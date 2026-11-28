use super::*;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, Barrier, mpsc};
use std::time::Duration;

fn clock(tick: &AtomicU64, begun: &AtomicU64, ahead: usize) -> (u64, Ripe) {
    (
        tick.load(Ordering::SeqCst),
        Ripe {
            started: begun.load(Ordering::SeqCst),
            finished: 0,
            ahead,
        },
    )
}

#[test]
fn blocked_growth_does_not_block_return_or_existing_piece() {
    let pool = Arc::new(Mutex::new(Pool::<u32>::new(2)));
    let gate = Arc::new(Mutex::new(()));
    let tick = Arc::new(AtomicU64::new(0));
    let held = lock(&pool).take(BLOCK / 2, 1, 0, || Some(1)).unwrap().0;
    let other = lock(&pool).take(BLOCK / 2, 1, 0, || None).unwrap().0;
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let worker = {
        let (pool, gate, tick, entered, release) = (
            pool.clone(),
            gate.clone(),
            tick.clone(),
            entered.clone(),
            release.clone(),
        );
        std::thread::spawn(move || {
            take_shared(
                &pool,
                &gate,
                BLOCK / 2,
                1,
                || (tick.load(Ordering::SeqCst), Ripe::ALL),
                || {
                    entered.wait();
                    release.wait();
                    Some(2)
                },
            )
        })
    };
    entered.wait();
    let lock_released = if let Ok(mut state) = pool.try_lock() {
        state.give(held, 0);
        state.reclaim(REST, 2, false);
        true
    } else {
        false
    };
    tick.store(REST, Ordering::SeqCst);
    let (sent, received) = mpsc::channel();
    let user = lock_released.then(|| {
        let (pool, gate, tick) = (pool.clone(), gate.clone(), tick.clone());
        std::thread::spawn(move || {
            let result = take_shared(
                &pool,
                &gate,
                BLOCK / 2,
                1,
                || (tick.load(Ordering::SeqCst), Ripe::ALL),
                || panic!("existing hit grew"),
            );
            sent.send(result).unwrap();
        })
    });
    let reused = user
        .as_ref()
        .map(|_| received.recv_timeout(Duration::from_secs(2)));
    release.wait();
    let grown = worker.join();
    if let Some(user) = user {
        user.join().unwrap();
    }
    assert!(lock_released, "driver growth held bookkeeping lock");
    let reused = reused
        .unwrap()
        .expect("existing hit waited for growth")
        .unwrap()
        .0;
    assert!(grown.unwrap().is_some());
    assert_eq!(reused, held);
    assert_ne!(reused, other);
}

#[test]
fn waiting_miss_rechecks_new_block_before_growth() {
    let pool = Arc::new(Mutex::new(Pool::<u32>::new(2)));
    let gate = Arc::new(Mutex::new(()));
    let calls = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let first = {
        let (pool, gate, calls, entered, release) = (
            pool.clone(),
            gate.clone(),
            calls.clone(),
            entered.clone(),
            release.clone(),
        );
        std::thread::spawn(move || {
            take_shared(
                &pool,
                &gate,
                BLOCK / 4,
                1,
                || (0, Ripe::ALL),
                || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    entered.wait();
                    release.wait();
                    Some(1)
                },
            )
            .unwrap()
            .0
        })
    };
    entered.wait();
    let second = {
        let (pool, gate, calls) = (pool.clone(), gate.clone(), calls.clone());
        std::thread::spawn(move || {
            take_shared(
                &pool,
                &gate,
                BLOCK / 4,
                1,
                || (0, Ripe::ALL),
                || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Some(2)
                },
            )
            .unwrap()
            .0
        })
    };
    release.wait();
    let a = first.join().unwrap();
    let b = second.join().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(a.block, b.block);
    assert!(a.start + a.len <= b.start || b.start + b.len <= a.start);
}

#[test]
fn failure_backoff_starts_at_completion_and_panic_can_retry() {
    let pool = Mutex::new(Pool::<u32>::new(2));
    let gate = Mutex::new(());
    let tick = AtomicU64::new(10);
    let calls = AtomicUsize::new(0);
    assert!(
        take_shared(
            &pool,
            &gate,
            4096,
            1,
            || (tick.load(Ordering::SeqCst), Ripe::ALL),
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                tick.store(1000, Ordering::SeqCst);
                None
            }
        )
        .is_none()
    );
    tick.store(1200, Ordering::SeqCst);
    assert!(
        take_shared(
            &pool,
            &gate,
            4096,
            1,
            || (tick.load(Ordering::SeqCst), Ripe::ALL),
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                Some(1)
            }
        )
        .is_none()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    tick.store(1500, Ordering::SeqCst);
    let panicked = std::panic::catch_unwind(|| {
        take_shared(
            &pool,
            &gate,
            4096,
            1,
            || (1500, Ripe::ALL),
            || -> Option<u32> { panic!("growth failed") },
        );
    });
    assert!(panicked.is_err());
    assert!(lock(&pool).blocks.is_empty());
    assert!(
        take_shared(
            &pool,
            &gate,
            4096,
            1,
            || (1500, Ripe::ALL),
            || {
                calls.fetch_add(1, Ordering::SeqCst);
                Some(1)
            }
        )
        .is_some()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn publication_stamps_fresh_submission_and_spare_failure_keeps_piece() {
    let pool = Mutex::new(Pool::<u32>::new(2));
    let gate = Mutex::new(());
    let tick = AtomicU64::new(1);
    let begun = AtomicU64::new(2);
    let calls = AtomicUsize::new(0);
    let result = take_shared(
        &pool,
        &gate,
        BLOCK / 2,
        1,
        || clock(&tick, &begun, 1),
        || {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                tick.store(20, Ordering::SeqCst);
                begun.store(7, Ordering::SeqCst);
                Some(1)
            } else {
                None
            }
        },
    );
    let (piece, _, _) = result.expect("spare failure lost reserved piece");
    let state = lock(&pool);
    assert_eq!(state.blocks[piece.block].as_ref().unwrap().made, 7);
    assert_eq!(state.retry_growth_at, 520);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn oldest_fitting_fallback_and_index_reuse_match_order() {
    let mut pool = Pool::<u32>::new(3);
    let a = pool.add(1, 9);
    let b = pool.add(2, 2);
    let c = pool.add(3, 5);
    let exact_age = |pool: &Pool<u32>| {
        let expected: BTreeSet<_> = pool
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, block)| block.as_ref().map(|block| (block.made, i)))
            .collect();
        assert_eq!(pool.age, expected);
    };
    exact_age(&pool);
    // Keep a live when reclaim drops b's now-empty block.
    pool.blocks[a].as_mut().unwrap().free.take(4096, 1).unwrap();
    let ripe = Ripe {
        started: 10,
        finished: 0,
        ahead: 1,
    };
    let (first, _, _) = pool.take_existing(BLOCK / 2, 1, 0, ripe).unwrap();
    assert_eq!(first.block, b);
    let (second, _, _) = pool.take_existing(BLOCK / 2, 1, 0, ripe).unwrap();
    assert_eq!(second.block, b);
    let (third, _, _) = pool.take_existing(BLOCK / 2, 1, 0, ripe).unwrap();
    assert_eq!(third.block, c);
    pool.give(first, 0);
    pool.give(second, 0);
    pool.reclaim(REST, 0, true);
    assert!(!pool.age.contains(&(2, b)));
    exact_age(&pool);
    let reused = pool.add(4, 12);
    assert_eq!(reused, b);
    assert!(pool.age.contains(&(12, b)));
    exact_age(&pool);
    // A ripe block wins before every older unripe block.
    let ripe = Ripe {
        started: 13,
        finished: 9,
        ahead: 1,
    };
    assert_eq!(
        pool.take_existing(BLOCK / 2, 1, REST, ripe)
            .unwrap()
            .0
            .block,
        a
    );
}

#[test]
fn same_age_tie_skips_fragmented_nonfit_without_changing_loser() {
    let mut pool = Pool::<u32>::new(2);
    let old = pool.add(1, 5);
    let next = pool.add(2, 5);
    let loser = pool.blocks[old].as_mut().unwrap();
    loser.free.take(BLOCK / 2, 1).unwrap();
    loser.free.take(4096, 1).unwrap();
    loser.used_to = BLOCK / 2 + 4096;
    let before = loser.free.0.clone();
    let used_to = loser.used_to;
    let ripe = Ripe {
        started: 6,
        finished: 0,
        ahead: 1,
    };
    let (piece, _, _) = pool.take_existing(BLOCK / 2, 65536, 0, ripe).unwrap();
    assert_eq!(piece.block, next);
    assert_eq!(piece.start, 0);
    assert_eq!(pool.blocks[old].as_ref().unwrap().free.0, before);
    assert_eq!(pool.blocks[old].as_ref().unwrap().used_to, used_to);
}
