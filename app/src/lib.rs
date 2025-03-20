use gfx::state::VxState;
use winit::{
    application::ApplicationHandler, dpi::LogicalSize, event::WindowEvent,
    event_loop::ActiveEventLoop,
};

pub struct VxApplication {
    app_name: &'static str,
    app_version: u32,
    title: String,
    size: LogicalSize<f32>,
    state: Option<VxState>,
}

impl VxApplication {
    pub fn new(
        app_name: &'static str,
        app_version: u32,
        title: &str,
        width: f32,
        height: f32,
    ) -> Self {
        Self {
            app_name,
            app_version,
            title: title.into(),
            size: LogicalSize::new(width, height),
            state: None,
        }
    }
}

impl ApplicationHandler for VxApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            self.state = Some(
                VxState::new(
                    self.app_name,
                    self.app_version,
                    &self.title,
                    &self.size,
                    event_loop,
                )
                .unwrap(),
            );
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let WindowEvent::CloseRequested = event {
            self.state = None;
            event_loop.exit();
        }
    }
}
