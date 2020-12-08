use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Grid<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T> Grid<T> {
    pub fn new(width: usize, height: usize, data: Vec<T>) -> Self {
        if width * height != data.len() {
            panic!(
                "width * height was {}, but must be equal to data.len(), which was {}",
                width * height,
                data.len()
            );
        }
        Self {
            width,
            height,
            data,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn get<I>(&self, idx: I) -> Option<&T>
    where
        GridIndex: From<I>,
    {
        let index: usize = self.linear_idx(GridIndex::from(idx))?;

        Some(&self.data[index])
    }

    pub fn get_mut<I>(&mut self, idx: I) -> Option<&mut T>
    where
        GridIndex: From<I>,
    {
        let index: usize = self.linear_idx(GridIndex::from(idx))?;

        Some(&mut self.data[index])
    }

    pub fn set<I>(&mut self, idx: I, item: T) -> Result<(), ()>
    where
        GridIndex: From<I>,
    {
        let index: usize = self.linear_idx(GridIndex::from(idx)).ok_or(())?;

        self.data[index] = item;

        Ok(())
    }

    fn linear_idx(&self, idx: GridIndex) -> Option<usize> {
        if idx.row() >= self.height() || idx.column() >= self.width() {
            None
        } else {
            Some(idx.row() * self.width() + idx.column())
        }
    }
}

impl<T, I> Index<I> for Grid<T>
where
    GridIndex: From<I>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        let index: GridIndex = GridIndex::from(index);

        let linear = self.linear_idx(index).unwrap_or_else(|| {
            panic!(
                "index out of bounds: ({},{}), but grid is of size ({},{})",
                index.row(),
                index.column(),
                self.width(),
                self.height()
            )
        });

        &self.data[linear]
    }
}

impl<T, I> IndexMut<I> for Grid<T>
where
    GridIndex: From<I>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index: GridIndex = GridIndex::from(index);

        let linear = self.linear_idx(index).unwrap_or_else(|| {
            panic!(
                "index out of bounds: ({},{}), but grid is of size ({},{})",
                index.row(),
                index.column(),
                self.width(),
                self.height()
            )
        });

        &mut self.data[linear]
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GridIndex {
    row: usize,
    column: usize,
}

impl GridIndex {
    pub fn new(row: usize, column: usize) -> Self {
        Self { row, column }
    }

    pub fn row(&self) -> usize {
        self.row
    }

    pub fn column(&self) -> usize {
        self.column
    }
}

impl From<(usize, usize)> for GridIndex {
    fn from((col, row): (usize, usize)) -> Self {
        GridIndex::new(row, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_test() {
        let grid = Grid::new(10, 10, (1..=100).collect());

        let mut counter = 0;
        for row in 0..grid.height() {
            for col in 0..grid.width() {
                counter += 1;
                assert_eq!(grid[(col, row)], counter);
            }
        }
    }
}
