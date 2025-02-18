use std::ops::{Add, Mul};
use std::error;
use std::fmt;

/// The error returned when both input sizes are not equal.
#[derive(Clone, Debug)]
pub struct SizeMismatch;
impl fmt::Display for SizeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Input Vecs were not the same size.")
    }
}
impl error::Error for SizeMismatch { }

/// A trait for any type that can have a dot product calculated from itself and another of Self.
pub trait Dot<T> {
    /// Returns the dot product of self and other.
    fn dot(&self, other: &Self) -> T;
}

/// An implementation of Dot for Vec.
/// 
/// Gives the dot product of two vectors.
/// 
/// # Examples
///
/// ```
/// # use semantica::vector::*;
/// assert_eq!(
///     vec![1, 2, 3].dot(&vec![4, 5, 6])
///         .unwrap(),
///     32
/// );
/// ```
/// 
/// # Panics
/// 
/// Will panic if input Vecs are not of the same size.
impl<T: Copy + Add<Output = T> + Mul<Output = T> + From<i8>> Dot<Result<T, SizeMismatch>> for Vec<T> {
    fn dot(&self, other: &Self) -> Result<T, SizeMismatch> {
        let len = self.len();

        if len != other.len() {
            return Err(SizeMismatch);
        }

        let mut output = 0.into();

        for i in 0..len {
            output = output + self[i] * other[i];
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn dot() {
        assert_eq!(
            vec![1, 2, 3].dot(&vec![4, 5, 6])
                .unwrap(),
            32
        );
    }

    #[test]
    #[should_panic]
    pub fn dot_panic() {
        assert_eq!(
            vec![1, 2, 3, 4].dot(&vec![4, 5, 6])
                .unwrap(),
            32
        );
    }
}