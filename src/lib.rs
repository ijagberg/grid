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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
