use glam::{Mat4, Vec3};

#[derive(Clone, Copy)]
pub struct Camera {
    pub model: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
}

#[derive(Clone, Copy)]
pub struct CameraRaw {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
}

impl Camera {
    pub fn from_position(pos: Vec3) -> Self {
        let model = Mat4::IDENTITY * Mat4::from_translation(pos);
        let view = Mat4::IDENTITY;
        let projection = Mat4::IDENTITY;

        Self {
            model,
            view,
            projection,
        }
    }

    pub fn translate(&mut self, vec: Vec3) {
        self.model *= Mat4::from_translation(vec);
    }

    pub fn to_raw(&self) -> CameraRaw {
        CameraRaw {
            model: self.model.to_cols_array_2d(),
            view: self.view.to_cols_array_2d(),
            projection: self.projection.to_cols_array_2d(),
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::from_position(Vec3::ZERO)
    }
}
