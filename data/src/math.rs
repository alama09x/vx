use std::ops::{Add, Div, Mul, Sub};

use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
}

impl Add for Aabb {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            min: self.min + rhs.min,
            max: self.max + rhs.max,
        }
    }
}

impl Sub for Aabb {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            min: self.min - rhs.min,
            max: self.max - rhs.max,
        }
    }
}

impl Mul for Aabb {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            min: self.min * rhs.min,
            max: self.max * rhs.max,
        }
    }
}

impl Div for Aabb {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            min: self.min / rhs.min,
            max: self.max / rhs.max,
        }
    }
}
