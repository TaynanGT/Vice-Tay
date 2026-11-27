//! Storage for the physical page numbers used by the full-size Series S
//! render targets. The title's fixed array has 16,384 slots, one per 64 KiB
//! page. Keep allocation and all three mapping paths on the same array.

use super::Site;

pub(super) const NATIVE_PAGES: u32 = 0x5000;

const REFERENCES: [Site; 4] = [
    Site {
        at: 0x5eb278d,
        what: "allocate render pages",
        bytes: &[0x4c, 0x8d, 0x05, 0x4c, 0x87, 0x73, 0x01],
    },
    Site {
        at: 0x2473fa1,
        what: "map render pages",
        bytes: &[0x4c, 0x8d, 0x1d, 0x38, 0x6f, 0x17, 0x05],
    },
    Site {
        at: 0x2474079,
        what: "map the next render pages",
        bytes: &[0x4c, 0x8d, 0x1d, 0x60, 0x6e, 0x17, 0x05],
    },
    Site {
        at: 0x24740ec,
        what: "map the remaining render pages",
        bytes: &[0x4c, 0x8d, 0x1d, 0xed, 0x6d, 0x17, 0x05],
    },
];

const ARRAY_BYTES: usize = NATIVE_PAGES as usize * size_of::<u64>();

fn reference_code(site: &Site, base: usize, pages: usize) -> Option<[u8; 7]> {
    let mut code: [u8; 7] = site.bytes.try_into().ok()?;
    let next = base.checked_add(site.at)?.checked_add(code.len())?;
    let displacement = i32::try_from(pages as i128 - next as i128).ok()?;
    code[3..].copy_from_slice(&displacement.to_le_bytes());
    Some(code)
}

fn allocation_bounds(
    base: usize,
    minimum: usize,
    maximum: usize,
    granularity: usize,
) -> Option<(usize, usize)> {
    let mut low = minimum;
    let mut high = maximum;
    for site in &REFERENCES {
        let next = base.checked_add(site.at)?.checked_add(site.bytes.len())?;
        low = low.max(next.saturating_add_signed(i32::MIN as isize));
        high = high.min(next.saturating_add(i32::MAX as usize));
    }
    let mask = granularity.checked_sub(1)?;
    let low = low.checked_add(mask)? & !mask;
    let high = (high.checked_add(1)? & !mask).checked_sub(1)?;
    (low.checked_add(ARRAY_BYTES - 1)? <= high).then_some((low, high))
}

#[cfg(windows)]
struct PageList(usize);

#[cfg(windows)]
impl PageList {
    fn allocate(base: usize) -> anyhow::Result<Self> {
        use anyhow::Context;
        use windows_sys::Win32::System::Memory::{
            MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE,
            MemExtendedParameterAddressRequirements, PAGE_READWRITE, VirtualAlloc2,
        };
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

        let mut system = SYSTEM_INFO::default();
        // SAFETY: system is writable for the duration of the call.
        unsafe { GetSystemInfo(&mut system) };
        let (low, high) = allocation_bounds(
            base,
            system.lpMinimumApplicationAddress as usize,
            system.lpMaximumApplicationAddress as usize,
            system.dwAllocationGranularity as usize,
        )
        .context("no address range in reach of all Series S render page references")?;

        let mut limits = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: low as *mut _,
            HighestEndingAddress: high as *mut _,
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut limits).cast();
        // SAFETY: a fresh allocation in this process; the parameter and its
        // aligned limits remain valid for the call.
        let address = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                std::ptr::null(),
                ARRAY_BYTES,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        };
        if address.is_null() {
            return Err(std::io::Error::last_os_error())
                .context("reserving the Series S render page list");
        }
        Ok(Self(address as usize))
    }
}

#[cfg(windows)]
impl Drop for PageList {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        // SAFETY: this reservation was never installed for a running title.
        unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
    }
}

/// The mapped title has not started. Each reference is checked before any
/// change. If a write fails, the caller stops startup.
#[cfg(windows)]
pub(super) unsafe fn relocate(base: usize, size: usize) -> anyhow::Result<()> {
    use anyhow::Context;
    use super::{Patched, matches, write};

    // Kept until process exit, like the title's original array. Other
    // title threads may still map pages after the entry point returns.
    static ACTIVE: std::sync::Mutex<Option<PageList>> = std::sync::Mutex::new(None);
    let mut active = ACTIVE.lock().unwrap_or_else(|err| err.into_inner());
    anyhow::ensure!(active.is_none(), "render page list is already installed");
    for site in &REFERENCES {
        // SAFETY: the caller has the title mapped and stopped.
        let result = unsafe { matches(base, size, site) };
        anyhow::ensure!(result == Patched::Done, "Series S 4K: {} ({result:?})", site.what);
    }
    let pages = PageList::allocate(base)?;
    for site in &REFERENCES {
        let code = reference_code(site, base, pages.0).context("render page list is out of reach")?;
        // SAFETY: the checked title is still mapped and has not started.
        let result = unsafe { write(base, size, site, &code) };
        anyhow::ensure!(result == Patched::Done, "Series S 4K: {} ({result:?})", site.what);
    }
    *active = Some(pages);
    tracing::info!(pages = NATIVE_PAGES, "Series S 4K: render page list expanded");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_and_mapping_use_the_same_larger_array() {
        let base = 0x1400_0000_usize;
        let array = base + 0x0780_0000;
        for site in &REFERENCES {
            let next = base + site.at + 7;
            let target = |bytes: &[u8]| next.checked_add_signed(i32::from_le_bytes(bytes[3..].try_into().unwrap()) as isize);
            assert_eq!(target(site.bytes), Some(base + 0x075e_aee0));
            let code = reference_code(site, base, array).unwrap();
            assert_eq!(&code[..3], &site.bytes[..3]);
            assert_eq!(target(&code), Some(array));
            assert!(reference_code(site, base, next + 0x8000_0000).is_none());
        }
        assert_eq!(ARRAY_BYTES, 0x0002_8000);
    }

    #[test]
    fn address_range_fits_every_reference_and_the_complete_reservation() {
        let base = 0x1_4000_0000;
        let granularity = 0x1_0000;
        let (low, high) = allocation_bounds(base, 0x1_0000, 0x7fff_ffff_ffff, granularity).unwrap();
        assert_eq!(low % granularity, 0);
        assert_eq!(high % granularity, granularity - 1);
        for site in &REFERENCES {
            assert!(reference_code(site, base, low).is_some());
            assert!(reference_code(site, base, high + 1 - ARRAY_BYTES).is_some());
        }
        assert!(allocation_bounds(base, low, low + granularity - 1, granularity).is_none());
        let bounds = allocation_bounds(base, low + 1, high - 1, granularity).unwrap();
        assert_eq!(bounds, (low + granularity, high - granularity));
    }
}
