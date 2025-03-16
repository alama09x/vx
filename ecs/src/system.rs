use thiserror::Error;

#[derive(Error, Debug)]
pub enum SystemError {}

pub struct System {
    params: Vec<Box<dyn SystemParam>>,
}

pub trait SystemParam {}
