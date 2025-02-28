/// A trait for any generic that can be converted from and to Vec<u8>.
pub trait ByteConversion {
    /// Converts Self to Vec<u8>.
    fn to_bytes(self) -> Vec<u8>;

    /// Converts Vec<u8> to Self.
    fn from_bytes(input: Vec<u8>) -> Self;
}
