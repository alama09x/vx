pub mod camera;
pub mod transform;

pub trait IntoBytes {
    fn to_bytes(&self) -> Vec<u8>;
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
