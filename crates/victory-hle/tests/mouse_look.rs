use std::time::Duration;

use victory_hle::modules::gameinput::{parse_env_flag, MouseButtons, MouseLook};

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected about {expected} (tolerance {tolerance}), got {actual}"
    );
}

// ---------------------------------------------------------------------------
// MouseLook
// ---------------------------------------------------------------------------

#[test]
fn reset_sets_both_axes_to_zero() {
    let mut look = MouseLook::default();
    let (x, y) = look.step(
        i32::MAX,
        -i32::MAX,
        Duration::from_millis(250),
        1000.0,
        false,
    );
    assert!(x > 0.0, "setup should create a positive X residual");
    assert!(y > 0.0, "setup should create a positive Y residual");

    look.reset();
    let (x, y) = look.step(0, 0, Duration::from_millis(1), 1.0, false);
    assert_eq!(x, 0.0);
    assert_eq!(y, 0.0);
}

#[test]
fn zero_delta_decays_by_half_after_one_half_life() {
    let mut look = MouseLook::default();
    look.reset();

    let (x, _) = look.step(
        i32::MAX,
        0,
        Duration::from_millis(250),
        1000.0,
        false,
    );
    assert_eq!(x, 1.0, "setup should saturate X to +1");

    let (x, y) = look.step(0, 0, Duration::from_millis(45), 1.0, false);
    assert_close(x, 0.5, 1e-5);
    assert_eq!(y, 0.0);
}

#[test]
fn repeated_zero_delta_steps_eventually_return_exact_zero() {
    let mut look = MouseLook::default();
    look.reset();

    let (x, y) = look.step(
        10_000,
        -10_000,
        Duration::from_millis(250),
        1.0,
        false,
    );
    assert!(x != 0.0);
    assert!(y != 0.0);

    for _ in 0..100 {
        let (x, y) = look.step(0, 0, Duration::from_millis(250), 1.0, false);
        if x == 0.0 && y == 0.0 {
            return;
        }
    }

    panic!("zero-delta steps did not eventually return exactly (0.0, 0.0)");
}

#[test]
fn horizontal_and_vertical_deltas_affect_separate_axes() {
    let mut look = MouseLook::default();
    look.reset();

    let (x, y) = look.step(1, -1, Duration::from_millis(1), 1.0, false);
    assert!(x > 0.0, "positive dx should yield positive X");
    assert!(y > 0.0, "negative dy without invert_y should yield positive Y");
}

#[test]
fn positive_dx_yields_positive_x() {
    let mut look = MouseLook::default();
    look.reset();

    let (x, _) = look.step(1, 0, Duration::from_millis(1), 1.0, false);
    assert!(x > 0.0);
}

#[test]
fn negative_dx_yields_negative_x() {
    let mut look = MouseLook::default();
    look.reset();

    let (x, _) = look.step(-1, 0, Duration::from_millis(1), 1.0, false);
    assert!(x < 0.0);
}

#[test]
fn negative_dy_without_invert_yields_positive_y() {
    let mut look = MouseLook::default();
    look.reset();

    let (_, y) = look.step(0, -1, Duration::from_millis(1), 1.0, false);
    assert!(y > 0.0);
}

#[test]
fn positive_dy_without_invert_yields_negative_y() {
    let mut look = MouseLook::default();
    look.reset();

    let (_, y) = look.step(0, 1, Duration::from_millis(1), 1.0, false);
    assert!(y < 0.0);
}

#[test]
fn negative_dy_with_invert_yields_negative_y() {
    let mut look = MouseLook::default();
    look.reset();

    let (_, y) = look.step(0, -1, Duration::from_millis(1), 1.0, true);
    assert!(y < 0.0);
}

#[test]
fn positive_dy_with_invert_yields_positive_y() {
    let mut look = MouseLook::default();
    look.reset();

    let (_, y) = look.step(0, 1, Duration::from_millis(1), 1.0, true);
    assert!(y > 0.0);
}

#[test]
fn one_pixel_at_low_sensitivity_is_nonzero() {
    let mut look = MouseLook::default();
    look.reset();

    let (x, _) = look.step(1, 0, Duration::from_millis(1), 0.3, false);
    assert!(x > 0.0);
    assert!(x != 0.0);
}

#[test]
fn higher_sensitivity_increases_response() {
    let mut low = MouseLook::default();
    low.reset();
    let (low_x, _) = low.step(1, 0, Duration::from_millis(1), 0.3, false);

    let mut normal = MouseLook::default();
    normal.reset();
    let (normal_x, _) = normal.step(1, 0, Duration::from_millis(1), 1.0, false);

    assert!(normal_x > low_x);
}

#[test]
fn dt_below_minimum_is_clamped_to_minimum() {
    let mut zero = MouseLook::default();
    zero.reset();
    let from_zero = zero.step(1, 0, Duration::ZERO, 1.0, false);

    let mut micro = MouseLook::default();
    micro.reset();
    let from_micro = micro.step(1, 0, Duration::from_micros(1), 1.0, false);

    let mut at_min = MouseLook::default();
    at_min.reset();
    let at_minimum = at_min.step(1, 0, Duration::from_millis(1), 1.0, false);

    assert_eq!(from_zero, from_micro);
    assert_eq!(from_zero, at_minimum);
}

#[test]
fn dt_above_maximum_is_clamped_to_maximum() {
    let mut just_above = MouseLook::default();
    just_above.reset();
    let from_251ms = just_above.step(1, 0, Duration::from_millis(251), 1.0, false);

    let mut long = MouseLook::default();
    long.reset();
    let from_one_second = long.step(1, 0, Duration::from_secs(1), 1.0, false);

    let mut at_max = MouseLook::default();
    at_max.reset();
    let at_maximum = at_max.step(1, 0, Duration::from_millis(250), 1.0, false);

    assert_eq!(from_251ms, from_one_second);
    assert_eq!(from_251ms, at_maximum);
}

#[test]
fn large_positive_dx_is_clamped_to_one() {
    let mut look = MouseLook::default();
    look.reset();

    let (x, _) = look.step(
        i32::MAX,
        0,
        Duration::from_millis(250),
        1000.0,
        false,
    );
    assert_eq!(x, 1.0);
}

#[test]
fn large_negative_dx_is_clamped_to_minus_one() {
    let mut look = MouseLook::default();
    look.reset();

    let (x, _) = look.step(
        -i32::MAX,
        0,
        Duration::from_millis(250),
        1000.0,
        false,
    );
    assert_eq!(x, -1.0);
}

#[test]
fn large_upward_dy_without_invert_y_is_clamped_to_one() {
    let mut look = MouseLook::default();
    look.reset();

    let (_, y) = look.step(
        0,
        -i32::MAX,
        Duration::from_millis(250),
        1000.0,
        false,
    );
    assert_eq!(y, 1.0);
}

#[test]
fn large_downward_dy_without_invert_y_is_clamped_to_minus_one() {
    let mut look = MouseLook::default();
    look.reset();

    let (_, y) = look.step(
        0,
        i32::MAX,
        Duration::from_millis(250),
        1000.0,
        false,
    );
    assert_eq!(y, -1.0);
}

#[test]
fn tiny_x_residual_is_snapped_when_dx_is_zero_even_with_nonzero_dy() {
    let mut look = MouseLook::default();
    look.reset();

    let half_life = Duration::from_millis(45);

    let (x, _) = look.step(
        i32::MAX,
        0,
        Duration::from_millis(250),
        1000.0,
        false,
    );
    assert_eq!(x, 1.0);

    for _ in 0..6 {
        let (x, _) = look.step(0, 0, half_life, 1.0, false);
        assert!(x > 0.0, "X residual should still be present before the final step");
    }

    let (x, y) = look.step(0, -1, half_life, 1.0, false);
    assert_eq!(x, 0.0);
    assert!(y > 0.0);
}

#[test]
fn tiny_y_residual_is_snapped_when_dy_is_zero_even_with_nonzero_dx() {
    let mut look = MouseLook::default();
    look.reset();

    let half_life = Duration::from_millis(45);

    let (_, y) = look.step(
        0,
        -i32::MAX,
        Duration::from_millis(250),
        1000.0,
        false,
    );
    assert_eq!(y, 1.0);

    for _ in 0..6 {
        let (_, y) = look.step(0, 0, half_life, 1.0, false);
        assert!(y > 0.0, "Y residual should still be present before the final step");
    }

    let (x, y) = look.step(1, 0, half_life, 1.0, false);
    assert_eq!(y, 0.0);
    assert!(x > 0.0);
}

// ---------------------------------------------------------------------------
// MouseButtons
// ---------------------------------------------------------------------------

#[test]
fn first_update_seeds_held_state_without_capturing() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([true, true, true], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, true, true], true),
        [false, false, false]
    );
}

#[test]
fn fresh_up_to_down_transition_is_captured() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [true, false, false]
    );
}

#[test]
fn captured_button_remains_true_while_held() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [true, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [true, false, false]
    );
}

#[test]
fn captured_button_remains_true_while_held_even_if_ineligible() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [true, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], false),
        [true, false, false]
    );
}

#[test]
fn release_makes_captured_button_false() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [true, false, false]
    );
    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
}

#[test]
fn press_while_ineligible_is_not_captured() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], false),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [false, false, false]
    );
}

#[test]
fn releasing_and_repressing_while_eligible_captures_button() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], false),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [true, false, false]
    );
}

#[test]
fn reset_clears_all_captured_state() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, true, true], true),
        [true, true, true]
    );

    buttons.reset();

    // All three are still held, but captured state must have been cleared.
    // eligible == false also prevents any new capture.
    assert_eq!(
        buttons.update([true, true, true], false),
        [false, false, false]
    );
}

#[test]
fn buttons_are_reported_in_lmb_rmb_mmb_order() {
    let mut buttons = MouseButtons::default();

    assert_eq!(
        buttons.update([false, false, false], true),
        [false, false, false]
    );
    assert_eq!(
        buttons.update([true, false, false], true),
        [true, false, false]
    );
    assert_eq!(
        buttons.update([false, true, false], true),
        [false, true, false]
    );
    assert_eq!(
        buttons.update([false, false, true], true),
        [false, false, true]
    );
    assert_eq!(
        buttons.update([true, true, true], true),
        [true, true, true]
    );
}

// ---------------------------------------------------------------------------
// parse_env_flag
// ---------------------------------------------------------------------------

#[test]
fn parse_env_flag_returns_true_for_documented_truthy_values() {
    for value in ["1", "true", "yes", "on"] {
        assert!(parse_env_flag(value), "expected true for {value}");
    }
}

#[test]
fn parse_env_flag_ignores_surrounding_whitespace_and_ascii_case() {
    for value in [" 1 ", "\ttrue\n", " YeS ", "oN", " TRUE "] {
        assert!(parse_env_flag(value), "expected true for {value}");
    }
}

#[test]
fn parse_env_flag_returns_false_for_documented_falsy_values() {
    for value in ["0", "false", "off", "no", "", "FALSE", "No"] {
        assert!(!parse_env_flag(value), "expected false for {value}");
    }
}

#[test]
fn parse_env_flag_returns_false_for_other_and_whitespace_only_values() {
    for value in ["2", "01", "enabled", "true!", " ", "\t\n"] {
        assert!(!parse_env_flag(value), "expected false for {value}");
    }
}
