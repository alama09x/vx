pub mod camera;
pub mod transform;

pub trait IntoBytes {
    fn to_bytes(&self) -> &[u8];
}

pub trait IntoBytesMut {
    fn to_bytes_mut(&mut self) -> &mut [u8];
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Left,
    Right,
    Down,
    Up,
    Back,
    Forward,
}
