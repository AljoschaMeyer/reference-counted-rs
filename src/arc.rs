// This code is adapted from the rust standard library Arc.

use base::borrow;
use base::cmp::Ordering;
use base::convert::{From, AsMut};
use base::fmt;
use base::hash::{Hash, Hasher};
use base::marker::{PhantomData, Unpin};
use base::mem;
use base::num::NonZeroUsize;
use base::ops::{Deref, DerefMut};
use base::ptr::{self, NonNull};
use base::sync::atomic;
use base::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};

use base::borrow::BorrowMut;

use base::prelude::v1::*;

use smart_pointer::{SmartPointer, IntoMut, SmartPointerMut};

use crate::ReferenceCounted;

/// A soft limit on the amount of references that may be made to an `Arc`.
///
/// Going above this limit will abort your program (although not
/// necessarily) at _exactly_ `MAX_REFCOUNT + 1` references.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

macro_rules! acquire {
    ($x:expr) => {
        atomic::fence(Acquire)
    };
}

/// A thread-safe reference-counted pointer.
pub struct Arc<T: ?Sized> {
    ptr: NonNull<ArcInner<T>>,
    phantom: PhantomData<ArcInner<T>>,
}

unsafe impl<T: ?Sized + Sync + Send> Send for Arc<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for Arc<T> {}

impl<T: ?Sized> Arc<T> {
    fn from_inner(ptr: NonNull<ArcInner<T>>) -> Self {
        Self { ptr, phantom: PhantomData }
    }
}

struct ArcInner<T: ?Sized> {
    strong: atomic::AtomicUsize,
    data: T,
}

impl<T: ?Sized> Arc<T> {
    #[inline]
    fn inner(&self) -> &ArcInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `ArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { self.ptr.as_ref() }
    }

    unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        // We are careful to *not* create a reference covering the "count" fields, as
        // this would alias with concurrent access to the reference counts (e.g. by `Weak`).
        unsafe { &mut (*this.ptr.as_ptr()).data }
    }

    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        // Destroy the data at this time, even though we may not free the box
        // allocation itself (there may still be weak pointers lying around).
        unsafe { ptr::drop_in_place(Self::get_mut_unchecked(self)) };
    }

    fn ptr(&self) -> *mut ArcInner<T> {
        self.ptr.as_ptr()
    }

    fn ref_count(&self) -> usize {
        self.inner().strong.load(SeqCst)
    }
}

impl<T: ?Sized> Clone for Arc<T> {
    /// Makes a clone of the `Arc` pointer.
    ///
    /// This creates another pointer to the same allocation, increasing the reference count.
    #[inline]
    fn clone(&self) -> Arc<T> {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let old_size = self.inner().strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arcs. If we don't do this the count can overflow
        // and users will use-after free. We racily saturate to `isize::MAX` on
        // the assumption that there aren't ~2 billion threads incrementing
        // the reference count at once. This branch will never be taken in
        // any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_size > MAX_REFCOUNT {
            panic!();
        }

        Self::from_inner(self.ptr)
    }
}

impl<T: ?Sized> Drop for Arc<T> {
    /// Drops the `Arc`.
    ///
    /// This will decrement the reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo  = Arc::new(Foo);
    /// let foo2 = Arc::clone(&foo);
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data.  Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // In particular, while the contents of an Arc are usually immutable, it's
        // possible to have interior writes to something like a Mutex<T>. Since a
        // Mutex is not acquired when it is deleted, we can't rely on its
        // synchronization logic to make writes in thread A visible to a destructor
        // running in thread B.
        //
        // Also note that the Acquire fence here could probably be replaced with an
        // Acquire load, which could improve performance in highly-contended
        // situations. See [2].
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // [2]: (https://github.com/rust-lang/rust/pull/41714)
        acquire!(self.inner().strong);

        unsafe {
            self.drop_slow();
        }
    }
}

impl<T: ?Sized> Deref for Arc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: ?Sized> borrow::Borrow<T> for Arc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> AsRef<T> for Arc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: ?Sized> SmartPointer<T> for Arc<T> {
    fn new(data: T) -> Arc<T> where T: Sized {
        let x: Box<_> = Box::new(ArcInner {
            strong: atomic::AtomicUsize::new(1),
            data,
        });
        Self::from_inner(Box::leak(x).into())
    }

    fn try_unwrap(this: Self) -> Result<T, Self> where T: Sized {
        if this.inner().strong.compare_exchange(1, 0, Relaxed, Relaxed).is_err() {
            return Err(this);
        }

        acquire!(this.inner().strong);

        unsafe {
            let elem = ptr::read(&this.ptr.as_ref().data);
            mem::forget(this);
            Ok(elem)
        }
    }
}

pub struct UniqueArc<T: ?Sized>(Arc<T>);

unsafe impl<T: ?Sized + Sync + Send> Send for UniqueArc<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for UniqueArc<T> {}

impl<T: ?Sized> Deref for UniqueArc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<T: ?Sized> borrow::Borrow<T> for UniqueArc<T> {
    fn borrow(&self) -> &T {
        self.0.borrow()
    }
}

impl<T: ?Sized> AsRef<T> for UniqueArc<T> {
    fn as_ref(&self) -> &T {
        self.0.as_ref()
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for UniqueArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for UniqueArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: ?Sized> fmt::Pointer for UniqueArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: ?Sized> SmartPointer<T> for UniqueArc<T> {
    fn new(data: T) -> Self where T: Sized {
        UniqueArc(Arc::new(data))
    }

    fn try_unwrap(this: Self) -> Result<T, Self> where T: Sized {
        let this = this.0;

        acquire!(this.inner().strong);

        unsafe {
            let elem = ptr::read(&this.ptr.as_ref().data);
            mem::forget(this);
            Ok(elem)
        }
    }
}


impl<T: ?Sized> DerefMut for UniqueArc<T> {
    fn deref_mut(&mut self) -> &mut T {
        // We know this to be uniquely owned
        unsafe { &mut (*self.0.ptr()).data }
    }
}

impl<T: ?Sized> BorrowMut<T> for UniqueArc<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T: ?Sized> AsMut<T> for UniqueArc<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

impl<T: ?Sized> SmartPointerMut<T> for UniqueArc<T> {}

impl<T: ?Sized> Into<Arc<T>> for UniqueArc<T> {
    fn into(self) -> Arc<T> {
        self.0
    }
}

impl<T: ?Sized> IntoMut<T> for Arc<T> {
    type MutablePointer = UniqueArc<T>;

    fn can_make_mut(this: &Self) -> bool {
        this.ref_count() == 1
    }

    unsafe fn into_mut_unchecked(this: Self) -> Self::MutablePointer {
        UniqueArc(this)
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

impl<T: ?Sized> ReferenceCounted<T> for Arc<T> {
    fn reference_count(this: &Self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(this.inner().strong.load(SeqCst)) }
    }
}

impl<T: Default> Default for Arc<T> {
    /// Creates a new `Arc<T>`, with the `Default` value for `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x: Arc<i32> = Default::default();
    /// assert_eq!(*x, 0);
    /// ```
    fn default() -> Arc<T> {
        Arc::new(Default::default())
    }
}

impl<T: Default> Default for UniqueArc<T> {
    /// Creates a new `UniqueArc<T>`, with the `Default` value for `T`.
    fn default() -> UniqueArc<T> {
        UniqueArc::new(Default::default())
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Arc<T> {
    /// Equality for two `Arc`s.
    ///
    /// Two `Arc`s are equal if their inner values are equal, even if they are
    /// stored in different allocation. This implementation does not check for
    /// pointer equality.
    #[inline]
    fn eq(&self, other: &Arc<T>) -> bool {
        (**self).eq(&**other)
    }

    /// Inequality for two `Arc`s.
    ///
    /// Two `Arc`s are unequal if their inner values are unequal. This implementation does not
    /// check for pointer equality.
    #[inline]
    fn ne(&self, other: &Arc<T>) -> bool {
        (**self).ne(&**other)
    }
}

impl<T: ?Sized + Eq> Eq for Arc<T> {}

impl<T: ?Sized + PartialEq> PartialEq for UniqueArc<T> {
    /// Equality for two `UniqueArc`s.
    ///
    /// Two `UniqueArc`s are equal if their inner values are equal, even if they are
    /// stored in different allocation. This implementation does not check for
    /// pointer equality.
    #[inline]
    fn eq(&self, other: &UniqueArc<T>) -> bool {
        (**self).eq(&**other)
    }

    /// Inequality for two `Arc`s.
    ///
    /// Two `Arc`s are unequal if their inner values are unequal. This implementation does not
    /// check for pointer equality.
    #[inline]
    fn ne(&self, other: &UniqueArc<T>) -> bool {
        (**self).ne(&**other)
    }
}

impl<T: ?Sized + Eq> Eq for UniqueArc<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Arc<T> {
    /// Partial comparison for two `Arc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    fn partial_cmp(&self, other: &Arc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Arc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    fn lt(&self, other: &Arc<T>) -> bool {
        *(*self) < *(*other)
    }

    /// 'Less than or equal to' comparison for two `Arc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    fn le(&self, other: &Arc<T>) -> bool {
        *(*self) <= *(*other)
    }

    /// Greater-than comparison for two `Arc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    fn gt(&self, other: &Arc<T>) -> bool {
        *(*self) > *(*other)
    }

    /// 'Greater than or equal to' comparison for two `Arc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    fn ge(&self, other: &Arc<T>) -> bool {
        *(*self) >= *(*other)
    }
}

impl<T: ?Sized + PartialOrd> PartialOrd for UniqueArc<T> {
    /// Partial comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    fn partial_cmp(&self, other: &UniqueArc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    fn lt(&self, other: &UniqueArc<T>) -> bool {
        *(*self) < *(*other)
    }

    /// 'Less than or equal to' comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    fn le(&self, other: &UniqueArc<T>) -> bool {
        *(*self) <= *(*other)
    }

    /// Greater-than comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    fn gt(&self, other: &UniqueArc<T>) -> bool {
        *(*self) > *(*other)
    }

    /// 'Greater than or equal to' comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    fn ge(&self, other: &UniqueArc<T>) -> bool {
        *(*self) >= *(*other)
    }
}

impl<T: ?Sized + Ord> Ord for Arc<T> {
    /// Comparison for two `Arc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    fn cmp(&self, other: &Arc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Ord> Ord for UniqueArc<T> {
    /// Comparison for two `UniqueArc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    fn cmp(&self, other: &UniqueArc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T> From<T> for Arc<T> {
    fn from(t: T) -> Self {
        Arc::new(t)
    }
}

impl<T> From<T> for UniqueArc<T> {
    fn from(t: T) -> Self {
        UniqueArc::new(t)
    }
}

impl<T: ?Sized + Hash> Hash for Arc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized + Hash> Hash for UniqueArc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized> Unpin for Arc<T> {}

impl<T: ?Sized> Unpin for UniqueArc<T> {}
