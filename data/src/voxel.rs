use std::fmt::Debug;

pub type VoxelId = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Voxel {
    Air = 0,
    Stone,
    Dirt,
    Grass,
}

impl Voxel {
    pub const VOXEL_COUNT: u8 = 4;
    pub const ALL: [Self; Self::VOXEL_COUNT as usize] =
        [Self::Air, Self::Stone, Self::Dirt, Self::Grass];

    pub const fn is_opaque(&self) -> bool {
        !matches!(self, Self::Air)
    }
}
