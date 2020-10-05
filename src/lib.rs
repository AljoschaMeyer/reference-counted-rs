#![no_std]
#![allow(unused_unsafe)]
// #![cfg_attr(feature = "unstable", coerce_unsized, dispatch_from_dyn)]
extern crate maybe_std as base;

use smart_pointer::IntoMut;

/// A smart pointer that keeps track of how many pointers refer to the same allocation and
/// exposes this information in its API.
pub trait ReferenceCounted<T: ?Sized>: IntoMut<T> + Clone {
    /// Get the number of owning pointers referring to the same allocation.
    ///
    /// Implementations must fulfill that `ReferenceCounted::reference_count(this) == 1` implies
    /// `IntoMut::con_make_mut(this) == true`.
    fn reference_count(this: &Self) -> usize;
}

#[cfg(feature = "arc")]
mod arc;
#[cfg(feature = "arc")]
pub use arc::*;

#[cfg(feature = "arc")]
mod rc;
#[cfg(feature = "arc")]
pub use rc::*;
