use std::{f32, ops::RangeInclusive};

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat4, Quat, Vec3};

use crate::{transform::Transform, IntoRaw, MoveDirection};

#[derive(Clone, Copy)]
pub struct Camera {
    transform: Transform,
    fov: f32,
    window_width: f32,
    window_height: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraRaw {
    pub view_projection: [[f32; 4]; 4],
}

impl Camera {
    pub fn new(transform: Transform, window_width: f32, window_height: f32) -> Self {
        Self {
            transform,
            fov: 45.0,
            window_width,
            window_height,
        }
    }

    const FOV_LIMIT_MIN: RangeInclusive<f32> = 1.0..=179.0;

    const SCROLL_SPEED: f32 = 3.0;

    pub fn zoom(&mut self, scroll: f32) {
        let degrees = scroll * 0.1 * Self::SCROLL_SPEED;
        if Self::FOV_LIMIT_MIN.contains(&(self.fov - degrees)) {
            println!("degs: {degrees}");
            self.fov -= degrees;
        }
    }

    const MOVE_SPEED: f32 = 3.0;

    pub fn move_in_direction(&mut self, direction: MoveDirection, delta_time: f32) {
        let speed = Self::MOVE_SPEED * delta_time;

        let remove_y = Vec3::X + Vec3::Z;
        let local_x = (self.transform.rotation * Vec3::X * remove_y).normalize() * speed;
        let local_z = (self.transform.rotation * Vec3::Z * remove_y).normalize() * speed;

        match direction {
            MoveDirection::Forward => self.transform.translation -= local_z,
            MoveDirection::Back => self.transform.translation += local_z,
            MoveDirection::Left => self.transform.translation -= local_x,
            MoveDirection::Right => self.transform.translation += local_x,
            MoveDirection::Up => self.transform.translation.y -= speed,
            MoveDirection::Down => self.transform.translation.y += speed,
        }
    }

    const YAW_SPEED: f32 = 0.004;
    const PITCH_SPEED: f32 = 0.004;
    const PITCH_LIMIT: f32 = f32::consts::FRAC_PI_2 - 0.01;

    pub fn rotate_by_mouse_movement(&mut self, dx: f32, dy: f32) {
        let dyaw = dx * Self::YAW_SPEED;
        let dpitch = -dy * Self::PITCH_SPEED;

        let (yaw, pitch, _roll) = self.transform.rotation.to_euler(EulerRot::YXZ);
        let yaw = yaw - dyaw;
        let pitch = (pitch - dpitch).clamp(-Self::PITCH_LIMIT, Self::PITCH_LIMIT);

        self.transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
    }
}

impl IntoRaw for Camera {
    type Raw = CameraRaw;

    fn to_raw(&self) -> Self::Raw {
        let view = self.transform.to_mat4().inverse();

        let projection = Mat4::perspective_rh(
            self.fov.to_radians(),
            self.window_width / self.window_height,
            0.1,
            100.0,
        );

        Self::Raw {
            view_projection: (projection * view).to_cols_array_2d(),
        }
    }
}
