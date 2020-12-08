use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Grid<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T> Grid<T> {
    pub fn new(width: usize, height: usize, data: Vec<T>) -> Self {
        if width * height == 0 {
            panic!("width * height cannot be 0");
        }
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

    pub fn area(&self) -> usize {
        self.width() * self.height()
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

    pub fn cell_iter<'a>(&'a self) -> CellIter<'a, T> {
        CellIter::new(0, self)
    }

    /// Return an iterator over the columns in `row`
    ///
    /// # Panics
    /// * If `row >= self.height()`
    pub fn row_iter<'a>(&'a self, row: usize) -> RowIter<'a, T> {
        if row >= self.height() {
            panic!(
                "row index out of bounds: the height is {} but the row index is {}",
                self.height(),
                row
            );
        }
        RowIter::new(row, 0, self)
    }

    /// Return an iterator over the rows in `column`
    ///
    /// # Panics
    /// * If `column >= self.height()`
    pub fn column_iter<'a>(&'a self, column: usize) -> ColIter<'a, T> {
        if column >= self.height() {
            panic!(
                "column index out of bounds: the height is {} but the column index is {}",
                self.height(),
                column
            );
        }
        ColIter::new(column, 0, self)
    }

    fn linear_idx(&self, idx: GridIndex) -> Option<usize> {
        if idx.row() >= self.height() || idx.column() >= self.width() {
            None
        } else {
            Some(idx.row() * self.width() + idx.column())
        }
    }
}

impl<T> IntoIterator for Grid<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
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

impl<T> Display for Grid<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // unwrap is safe here because we can't create a grid with length 0
        let max_length = self.cell_iter().map(|c| c.to_string().len()).max().unwrap();
        let output = (0..self.height())
            .map(|r| {
                (0..self.width())
                    .map(|c| {
                        let elem = &self[(c, r)].to_string();
                        let padding = std::iter::repeat(" ".to_string())
                            .take(max_length - elem.len())
                            .collect::<String>();
                        format!("{}{}", padding, self[(c, r)].to_string())
                    })
                    .collect::<Vec<String>>()
                    .join(" ")
            })
            .collect::<Vec<String>>()
            .join("\n");

        write!(f, "{}", output)
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

pub struct CellIter<'a, T> {
    current: usize,
    grid: &'a Grid<T>,
}

impl<'a, T> CellIter<'a, T> {
    fn new(current: usize, grid: &'a Grid<T>) -> Self {
        Self { current, grid }
    }
}

impl<'a, T> Iterator for CellIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        if current >= self.grid.data.len() {
            None
        } else {
            self.current = current + 1;
            let item = &self.grid.data[current];
            Some(item)
        }
    }
}

pub struct RowIter<'a, T> {
    row: usize,
    current_col: usize,
    grid: &'a Grid<T>,
}

impl<'a, T> RowIter<'a, T> {
    fn new(row: usize, current_col: usize, grid: &'a Grid<T>) -> Self {
        Self {
            row,
            current_col,
            grid,
        }
    }
}

impl<'a, T> Iterator for RowIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let current_col = self.current_col;
        if current_col == self.grid.width() {
            None
        } else {
            self.current_col = current_col + 1;
            let item = &self.grid[(current_col, self.row)];
            Some(item)
        }
    }
}

impl<'a, T> DoubleEndedIterator for RowIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let current_col = self.current_col;
        if current_col == 0 {
            None
        } else {
            self.current_col = current_col - 1;
            let item = &self.grid[(current_col, self.row)];
            Some(item)
        }
    }
}

pub struct ColIter<'a, T> {
    col: usize,
    current_row: usize,
    grid: &'a Grid<T>,
}

impl<'a, T> ColIter<'a, T> {
    fn new(col: usize, current_row: usize, grid: &'a Grid<T>) -> Self {
        Self {
            col,
            current_row,
            grid,
        }
    }
}

impl<'a, T> Iterator for ColIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let current_row = self.current_row;
        if current_row == self.grid.height() {
            None
        } else {
            self.current_row = current_row + 1;
            let item = &self.grid[(self.col, current_row)];
            Some(item)
        }
    }
}

impl<'a, T> DoubleEndedIterator for ColIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let current_row = self.current_row;
        if current_row == 0 {
            None
        } else {
            self.current_row = current_row - 1;
            let item = &self.grid[(self.col, current_row)];
            Some(item)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    fn example_grid() -> Grid<u32> {
        let grid = Grid::new(10, 10, (1..=100).collect());

        println!("{}", grid);

        grid
    }

    #[test]
    fn index_test() {
        let grid = example_grid();

        let mut counter = 0;
        for row in 0..grid.height() {
            for col in 0..grid.width() {
                counter += 1;
                assert_eq!(grid[(col, row)], counter);
            }
        }
    }

    #[test]
    fn row_iter_test() {
        let grid = example_grid();

        let actual_items_in_row: Vec<u32> = grid.row_iter(2).copied().collect();

        assert_eq!(
            actual_items_in_row,
            vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        );
    }

    #[test]
    fn col_iter_test() {
        let grid = example_grid();

        let actual_items_in_col: Vec<u32> = grid.column_iter(2).copied().collect();

        assert_eq!(
            actual_items_in_col,
            vec![3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
        );
    }
}
