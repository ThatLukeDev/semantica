/// A trait for any generic that can be converted from and to Vec<u8>.
pub trait ByteConversion {
    /// Converts Self to Vec<u8>.
    fn to_bytes(self) -> Vec<u8>;

    /// Converts Vec<u8> to Self.
    fn from_bytes(input: Vec<u8>) -> Self;
}

macro_rules! default_impl_ByteConversion {
    ($($t:ty),+) => {
        $(impl ByteConversion for $t {
            fn to_bytes(self) -> Vec<u8> {
                self.to_be_bytes().to_vec()
            }

            fn from_bytes(input: Vec<u8>) -> Self {
                Self::from_be_bytes(input.try_into().unwrap())
            }
        })+
    }
}

default_impl_ByteConversion!(usize, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

impl ByteConversion for String {
    fn to_bytes(self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }

    fn from_bytes(input: Vec<u8>) -> Self {
        String::from_utf8(input).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_default_impl_ByteConversion {
        ($($t:tt),+) => {
            $(let val: $t = $t::MAX; assert_eq!(
                    $t::from_bytes(val.to_bytes()),
                    val
            );)+
        }
    }

    #[test]
    fn num_types() {
        test_default_impl_ByteConversion!(usize, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);
    }

    #[test]
    fn string_types() {
        let str = "Hello World!";
        assert_eq!(
            String::from_bytes(str.to_string().to_bytes()),
            str.to_string()
        );
    }
}
