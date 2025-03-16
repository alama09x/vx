mod component;
mod manager;
mod query;
mod resource;
mod schedule;
mod system;

pub use component::*;
pub use manager::*;
pub use query::*;
pub use resource::*;
pub use schedule::*;
pub use system::*;

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
