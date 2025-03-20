use gfx::state::VxState;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes},
};

pub struct VxApplication {
    app_name: &'static str,
    app_version: u32,
    title: String,
    size: LogicalSize<f32>,
    window: Option<Window>,
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
            window: None,
            state: None,
        }
    }
}

impl ApplicationHandler for VxApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.window = Some(
                event_loop
                    .create_window(
                        WindowAttributes::default()
                            .with_title(&self.title)
                            .with_inner_size(self.size),
                    )
                    .unwrap(),
            );
        }

        if self.state.is_none() {
            self.state = Some(
                VxState::new(
                    self.app_name,
                    self.app_version,
                    self.window.as_ref().unwrap(),
                )
                .unwrap(),
            );
        }
        println!("Initialized");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let WindowEvent::CloseRequested = event {
            unsafe {
                if let Some(state) = self.state.take() {
                    state.device.device_wait_idle().unwrap();
                    drop(state);
                }
            }
            self.window.take();
            event_loop.exit();
        }
    }
}
