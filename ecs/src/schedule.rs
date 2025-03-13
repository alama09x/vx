#[derive(Debug)]
pub enum Schedule {
    PreStartup,
    Startup,
    PostStartup,
    Update,
    Cleanup,
}
