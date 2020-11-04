// This code is adapted from the rust standard library Rc.

use base::alloc::{dealloc, Layout};
use base::borrow;
use base::cell::Cell;
use base::cmp::Ordering;
use base::convert::{From, AsMut};
use base::fmt;
use base::hash::{Hash, Hasher};
use base::marker::{PhantomData, Unpin};
use base::mem;
use base::num::NonZeroUsize;
use base::ops::{Deref, DerefMut};
use base::ptr::{self, NonNull};

use base::borrow::BorrowMut;

use base::prelude::v1::*;

use smart_pointer::{SmartPointer, IntoMut, SmartPointerMut};

use crate::ReferenceCounted;

/// A non-thread-safe reference-counted pointer.
pub struct Rc<T: ?Sized> {
    ptr: NonNull<RcBox<T>>,
    phantom: PhantomData<RcBox<T>>,
}

struct RcBox<T: ?Sized> {
    strong: Cell<usize>,
    data: T,
}

impl<T: ?Sized> Rc<T> {
    fn from_inner(ptr: NonNull<RcBox<T>>) -> Self {
        Self { ptr, phantom: PhantomData }
    }

    #[inline]
    fn inner(&self) -> &RcBox<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid.
        unsafe { self.ptr.as_ref() }
    }

    fn ptr(&self) -> *mut RcBox<T> {
        self.ptr.as_ptr()
    }

    fn ref_count(&self) -> usize {
        self.inner().strong.get()
    }

    #[inline]
    fn inc_strong(&self) {
        let strong = self.ref_count();

        // We want to abort on overflow instead of dropping the value.
        // The reference count will never be zero when this is called;
        // nevertheless, we insert an abort here to hint LLVM at
        // an otherwise missed optimization.
        if strong == 0 || strong == usize::MAX {
            panic!();
        }
        self.inner().strong.set(strong + 1);
    }

    #[inline]
    fn dec_strong(&self) {
        self.inner().strong.set(self.ref_count() - 1);
    }
}

impl<T: ?Sized> Clone for Rc<T> {
    /// Makes a clone of the `Rc` pointer.
    ///
    /// This creates another pointer to the same allocation, increasing the reference count.
    #[inline]
    fn clone(&self) -> Rc<T> {
        self.inc_strong();
        Self::from_inner(self.ptr)
    }
}

impl<T: ?Sized> Drop for Rc<T> {
    /// Drops the `Rc`.
    ///
    /// This will decrement the reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Rc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo  = Rc::new(Foo);
    /// let foo2 = Rc::clone(&foo);
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.dec_strong();
            if self.ref_count() == 0 {
                // destroy the contained object
                ptr::drop_in_place(self.ptr.as_mut());

                dealloc(self.ptr().cast(), Layout::for_value(self.ptr.as_ref()));
            }
        }
    }
}

impl<T: ?Sized> Deref for Rc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: ?Sized> borrow::Borrow<T> for Rc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> AsRef<T> for Rc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for Rc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: ?Sized> SmartPointer<T> for Rc<T> {
    fn new(data: T) -> Rc<T> where T: Sized {
        Self::from_inner(
            Box::leak(Box::new(RcBox { strong: Cell::new(1), data })).into(),
        )
    }

    fn try_unwrap(this: Self) -> Result<T, Self> where T: Sized {
        if Rc::ref_count(&this) == 1 {
            unsafe {
                let val = ptr::read(&*this); // copy the contained object
                dealloc(this.ptr().cast(), Layout::for_value(this.ptr.as_ref()));
                mem::forget(this);
                Ok(val)
            }
        } else {
            Err(this)
        }
    }
}

pub struct UniqueRc<T: ?Sized>(Rc<T>);

unsafe impl<T: ?Sized + Sync + Send> Send for UniqueRc<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for UniqueRc<T> {}

impl<T: ?Sized> Deref for UniqueRc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<T: ?Sized> borrow::Borrow<T> for UniqueRc<T> {
    fn borrow(&self) -> &T {
        self.0.borrow()
    }
}

impl<T: ?Sized> AsRef<T> for UniqueRc<T> {
    fn as_ref(&self) -> &T {
        self.0.as_ref()
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for UniqueRc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for UniqueRc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: ?Sized> fmt::Pointer for UniqueRc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: ?Sized> SmartPointer<T> for UniqueRc<T> {
    fn new(data: T) -> Self where T: Sized {
        UniqueRc(Rc::new(data))
    }

    fn try_unwrap(this: Self) -> Result<T, Self> where T: Sized {
        let this = this.0;

        unsafe {
            let elem = ptr::read(&this.ptr.as_ref().data);
            dealloc(this.ptr().cast(), Layout::for_value(this.ptr.as_ref()));
            mem::forget(this);
            Ok(elem)
        }
    }
}


impl<T: ?Sized> DerefMut for UniqueRc<T> {
    fn deref_mut(&mut self) -> &mut T {
        // We know this to be uniquely owned
        unsafe { &mut (*self.0.ptr()).data }
    }
}

impl<T: ?Sized> BorrowMut<T> for UniqueRc<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T: ?Sized> AsMut<T> for UniqueRc<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T: ?Sized> SmartPointerMut<T> for UniqueRc<T> {}

impl<T: ?Sized> Into<Rc<T>> for UniqueRc<T> {
    fn into(self) -> Rc<T> {
        self.0
    }
}

impl<T: ?Sized> IntoMut<T> for Rc<T> {
    type MutablePointer = UniqueRc<T>;

    fn can_make_mut(this: &Self) -> bool {
        this.ref_count() == 1
    }

    unsafe fn into_mut_unchecked(this: Self) -> Self::MutablePointer {
        UniqueRc(this)
    }

    /// Obtain a mutable reference to the wrapped value without performing runtime checks for
    /// upholding any invariants.
    ///
    /// Safety: Calling this is safe if and only if `can_make_mut` returns true.
    unsafe fn get_mut_unchecked(this: &Self) -> &mut T {
        // We are careful to *not* create a reference covering the "count" fields, as
        // this would alias with concurrent access to the reference counts (e.g. by `Weak`).
        unsafe { &mut (*this.ptr.as_ptr()).data }
    }
}

impl<T: ?Sized> ReferenceCounted<T> for Rc<T> {
    fn reference_count(this: &Self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(this.ref_count()) }
    }
}

impl<T: Default> Default for Rc<T> {
    /// Creates a new `Rc<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Rc;
    ///
    /// let x: Rc<i32> = Default::default();
    /// assert_eq!(*x, 0);
    /// ```
    fn default() -> Rc<T> {
        Rc::new(Default::default())
    }
}

impl<T: Default> Default for UniqueRc<T> {
    /// Creates a new `UniqueRc<T>`, with the `Default` value for `T`.
    fn default() -> UniqueRc<T> {
        UniqueRc::new(Default::default())
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Rc<T> {
    /// Equality for two `Rc`s.
    ///
    /// Two `Rc`s are equal if their inner values are equal, even if they are
    /// stored in different allocation. This implementation does not check for
    /// pointer equality.
    #[inline]
    fn eq(&self, other: &Rc<T>) -> bool {
        (**self).eq(&**other)
    }

    /// Inequality for two `Rc`s.
    ///
    /// Two `Rc`s are unequal if their inner values are unequal. This implementation does not
    /// check for pointer equality.
    #[inline]
    fn ne(&self, other: &Rc<T>) -> bool {
        (**self).ne(&**other)
    }
}

impl<T: ?Sized + Eq> Eq for Rc<T> {}

impl<T: ?Sized + PartialEq> PartialEq for UniqueRc<T> {
    /// Equality for two `UniqueRc`s.
    ///
    /// Two `UniqueRc`s are equal if their inner values are equal, even if they are
    /// stored in different allocation. This implementation does not check for
    /// pointer equality.
    #[inline]
    fn eq(&self, other: &UniqueRc<T>) -> bool {
        (**self).eq(&**other)
    }

    /// Inequality for two `Rc`s.
    ///
    /// Two `Rc`s are unequal if their inner values are unequal. This implementation does not
    /// check for pointer equality.
    #[inline]
    fn ne(&self, other: &UniqueRc<T>) -> bool {
        (**self).ne(&**other)
    }
}

impl<T: ?Sized + Eq> Eq for UniqueRc<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Rc<T> {
    /// Partial comparison for two `Rc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    fn partial_cmp(&self, other: &Rc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Rc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    fn lt(&self, other: &Rc<T>) -> bool {
        *(*self) < *(*other)
    }

    /// 'Less than or equal to' comparison for two `Rc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    fn le(&self, other: &Rc<T>) -> bool {
        *(*self) <= *(*other)
    }

    /// Greater-than comparison for two `Rc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    fn gt(&self, other: &Rc<T>) -> bool {
        *(*self) > *(*other)
    }

    /// 'Greater than or equal to' comparison for two `Rc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    fn ge(&self, other: &Rc<T>) -> bool {
        *(*self) >= *(*other)
    }
}

impl<T: ?Sized + PartialOrd> PartialOrd for UniqueRc<T> {
    /// Partial comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    fn partial_cmp(&self, other: &UniqueRc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    fn lt(&self, other: &UniqueRc<T>) -> bool {
        *(*self) < *(*other)
    }

    /// 'Less than or equal to' comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    fn le(&self, other: &UniqueRc<T>) -> bool {
        *(*self) <= *(*other)
    }

    /// Greater-than comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    fn gt(&self, other: &UniqueRc<T>) -> bool {
        *(*self) > *(*other)
    }

    /// 'Greater than or equal to' comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    fn ge(&self, other: &UniqueRc<T>) -> bool {
        *(*self) >= *(*other)
    }
}

impl<T: ?Sized + Ord> Ord for Rc<T> {
    /// Comparison for two `Rc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    fn cmp(&self, other: &Rc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Ord> Ord for UniqueRc<T> {
    /// Comparison for two `UniqueRc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    fn cmp(&self, other: &UniqueRc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T> From<T> for Rc<T> {
    fn from(t: T) -> Self {
        Rc::new(t)
    }
}

impl<T> From<T> for UniqueRc<T> {
    fn from(t: T) -> Self {
        UniqueRc::new(t)
    }
}

impl<T: ?Sized + Hash> Hash for Rc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized + Hash> Hash for UniqueRc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized> Unpin for Rc<T> {}

impl<T: ?Sized> Unpin for UniqueRc<T> {}
