use std::{any::Any, mem};

use bytemuck::Pod;

pub mod camera;
pub mod transform;

pub trait IntoRaw {
    type Raw: Pod;
    fn to_raw(&self) -> Self::Raw;
}

pub trait DynIntoRaw {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn raw_size(&self) -> usize;
    fn to_bytes(&self) -> Vec<u8>;
}

impl<R: Pod, T: 'static + IntoRaw<Raw = R>> DynIntoRaw for T {
    fn raw_size(&self) -> usize {
        mem::size_of::<T::Raw>()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn to_bytes(&self) -> Vec<u8> {
        bytemuck::cast_slice(&[self.to_raw()]).to_vec()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MoveDirection {
    Left,
    Right,
    Down,
    Up,
    Back,
    Forward,
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
