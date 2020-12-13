use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Grid<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T> Grid<T> {
    /// Construct a new Grid
    ///
    /// # Panics
    /// * If `width * height == 0`
    /// * If `width * height != data.len()`
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

    fn data(&self) -> &Vec<T> {
        &self.data
    }

    fn set_height(&mut self, v: usize) {
        self.height = v;
    }

    fn set_width(&mut self, v: usize) {
        self.width = v;
    }

    /// Returns the width (number of columns) of the grid
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height (number of rows) of the grid
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the area (number of columns * number of rows) of the grid
    pub fn area(&self) -> usize {
        self.width() * self.height()
    }

    /// Attempts to get a reference to the element at `idx`
    ///
    /// Returns `None` if `idx` is out of bounds
    pub fn get<I>(&self, idx: I) -> Option<&T>
    where
        GridIndex: From<I>,
    {
        let index: usize = self.linear_idx(GridIndex::from(idx)).ok()?;

        Some(&self.data()[index])
    }

    /// Attempts to get a mutable reference to the element at `idx`
    ///
    /// Returns `None` if `idx` is out of bounds
    pub fn get_mut<I>(&mut self, idx: I) -> Option<&mut T>
    where
        GridIndex: From<I>,
    {
        let index: usize = self.linear_idx(GridIndex::from(idx)).ok()?;

        Some(&mut self.data[index])
    }

    pub fn cell_iter<'a>(&'a self) -> CellIter<'a, T> {
        CellIter::new(0, self)
    }

    /// Return an iterator over the columns in `row`
    ///
    /// # Panics
    /// * If `row >= self.height()`
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(10, 10, (1..=100).collect());
    /// let items_in_row_2: Vec<u32> = grid.row_iter(2).cloned().collect();
    /// assert_eq!(items_in_row_2, vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]);
    /// ```
    pub fn row_iter<'a>(&'a self, row: usize) -> RowIter<'a, T> {
        self.panic_if_row_out_of_bounds(row);
        RowIter::new(row, 0, self)
    }

    /// Return an iterator over the rows in `column`
    ///
    /// # Panics
    /// * If `column >= self.height()`
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(10, 10, (1..=100).collect());
    /// let items_in_column_2: Vec<u32> = grid.column_iter(2).cloned().collect();
    /// assert_eq!(items_in_column_2, vec![3, 13, 23, 33, 43, 53, 63, 73, 83, 93]);
    /// ```
    pub fn column_iter<'a>(&'a self, column: usize) -> ColIter<'a, T> {
        self.panic_if_column_out_of_bounds(column);
        ColIter::new(column, 0, self)
    }

    /// Add a row at index `row`, moving all other rows backwards (row n becomes row n+1 and so on)
    ///
    /// # Panics
    /// * If `row_contents.len() != self.width()`
    /// * If `row >= self.height()`
    pub fn add_row(&mut self, row: usize, row_contents: Vec<T>) {
        if row_contents.len() != self.width() {
            panic!(
                "invalid length of row: was {}, should be {}",
                row_contents.len(),
                self.width()
            );
        }

        self.panic_if_row_out_of_bounds(row);

        let start_idx = self.linear_idx(GridIndex::new(row, 0)).unwrap();

        for (elem, idx) in row_contents.into_iter().zip(start_idx..) {
            self.data.insert(idx, elem);
        }

        self.set_height(self.height() + 1);
    }

    /// Remove row at `row`, shifting all rows with higher indices "upward" (row 4 becomes row 3 etc.)
    ///
    /// # Panics
    /// * If `self.height() == 1`
    /// * If `row >= self.height()`
    pub fn remove_row(&mut self, row: usize) {
        if self.height() == 1 {
            panic!("can't remove row if height is 1");
        }
        self.panic_if_row_out_of_bounds(row);

        let start_idx = self.linear_idx(GridIndex::new(row, 0)).unwrap();

        self.data.drain(start_idx..start_idx + self.width());
        self.set_height(self.height() - 1);
    }

    /// Add a column at index `column`, moving all other columns backwards (column n becomes column n+1 and so on)
    ///
    /// # Panics
    /// * If `column_contents.len() != self.height()`
    /// * If `column >= self.width()`
    pub fn add_column(&mut self, column: usize, column_contents: Vec<T>) {
        if column_contents.len() != self.height() {
            panic!(
                "invalid length of column: was {}, should be {}",
                column_contents.len(),
                self.height()
            );
        }

        self.panic_if_column_out_of_bounds(column);

        let indices: Vec<usize> = (0..column_contents.len())
            .map(|row| self.linear_idx(GridIndex::new(row, column)).unwrap())
            .collect();

        for (elem, idx) in column_contents.into_iter().zip(indices.into_iter()).rev() {
            self.data.insert(idx, elem);
        }

        self.set_width(self.width() + 1);
    }

    /// Remove column at `column`, shifting all columns with higher indices "left" (column 4 becomes column 3 etc.)
    ///
    /// # Panics
    /// * If `self.width() == 1`
    /// * If `column >= self.width()`
    pub fn remove_column(&mut self, column: usize) {
        if self.width() == 1 {
            panic!("can't remove column if width is 1");
        }
        self.panic_if_column_out_of_bounds(column);

        let indices: Vec<usize> = (0..self.height())
            .map(|row| self.linear_idx(GridIndex::new(row, column)).unwrap())
            .collect();

        for idx in indices.into_iter().rev() {
            self.data.remove(idx);
        }

        self.set_width(self.width() - 1);
    }

    fn linear_idx(&self, idx: GridIndex) -> Result<usize, LinearIndexError> {
        if idx.row() >= self.height() {
            Err(LinearIndexError::RowTooHigh)
        } else if idx.column() >= self.width() {
            Err(LinearIndexError::ColumnTooHigh)
        } else {
            Ok(idx.row() * self.width() + idx.column())
        }
    }

    fn panic_if_row_out_of_bounds(&self, row: usize) {
        if row >= self.height() {
            panic!(
                "row index out of bounds: the height is {} but the row index is {}",
                self.height(),
                row
            );
        }
    }

    fn panic_if_column_out_of_bounds(&self, column: usize) {
        if column >= self.width() {
            panic!(
                "column index out of bounds: the width is {} but the column index is {}",
                self.width(),
                column
            );
        }
    }
}

impl<T> Grid<T>
where
    T: Clone,
{
    /// Rotate the grid clockwise 90 degrees, cloning the data into a new grid
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, vec!['a', 'b', 'c', 'd']);
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    /// 
    /// let cw = grid.rotate_cw();
    /// assert_eq!(cw, Grid::new(2, 2, vec!['c', 'a', 'd', 'b']));
    /// println!("{}", cw);
    /// // prints:
    /// // c a
    /// // d b
    /// ```
    pub fn rotate_cw(&self) -> Self {
        let mut rotated_data = Vec::with_capacity(self.area());

        for column in 0..self.width() {
            for row in (0..self.height()).rev() {
                rotated_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.height(), self.width(), rotated_data)
    }

    /// Rotate the grid counter-clockwise 90 degrees, cloning the data into a new grid
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, vec!['a', 'b', 'c', 'd']);
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    /// 
    /// let ccw = grid.rotate_ccw();
    /// assert_eq!(ccw, Grid::new(2, 2, vec!['b', 'd', 'a', 'c']));
    /// println!("{}", ccw);
    /// // prints:
    /// // b d
    /// // a c
    /// ```
    pub fn rotate_ccw(&self) -> Self {
        let mut rotated_data = Vec::with_capacity(self.area());

        for column in (0..self.width()).rev() {
            for row in 0..self.height() {
                rotated_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.height(), self.width(), rotated_data)
    }

    /// Flip the grid horizontally, so that the first column becomes the last
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, vec!['a', 'b', 'c', 'd']);
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    /// 
    /// let hori = grid.flip_horizontally();
    /// assert_eq!(hori, Grid::new(2, 2, vec!['b', 'a', 'd', 'c']));
    /// println!("{}", hori);
    /// // prints:
    /// // b a
    /// // d c
    /// ```
    pub fn flip_horizontally(&self) -> Self {
        let mut flipped_data = Vec::with_capacity(self.area());

        for row in 0..self.height() {
            for column in (0..self.width()).rev() {
                flipped_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.width(), self.height(), flipped_data)
    }

    /// Flip the grid vertically, so that the first row becomes the last
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, vec!['a', 'b', 'c', 'd']);
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    /// 
    /// let vert = grid.flip_vertically();
    /// assert_eq!(vert, Grid::new(2, 2, vec!['c', 'd', 'a', 'b']));
    /// println!("{}", vert);
    /// // prints:
    /// // c d
    /// // a b
    /// ```
    pub fn flip_vertically(&self) -> Self {
        let mut flipped_data = Vec::with_capacity(self.area());

        for row in (0..self.height()).rev() {
            for column in 0..self.width() {
                flipped_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.width(), self.height(), flipped_data)
    }
}

impl<T> Grid<T>
where
    T: Default,
{
    /// Create a grid filled with default values
    pub fn new_default(width: usize, height: usize) -> Grid<T> {
        let data = (0..width * height).map(|_| T::default()).collect();
        Self::new(width, height, data)
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

        let linear = self.linear_idx(index).unwrap();

        &self.data[linear]
    }
}

impl<T, I> IndexMut<I> for Grid<T>
where
    GridIndex: From<I>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index: GridIndex = GridIndex::from(index);

        let linear = self.linear_idx(index).unwrap();

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

#[derive(Debug, Copy, Clone)]
pub enum LinearIndexError {
    RowTooHigh,
    ColumnTooHigh,
}

impl Display for LinearIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output = match self {
            LinearIndexError::RowTooHigh => "row index is too high",
            LinearIndexError::ColumnTooHigh => "column index is too high",
        };

        write!(f, "{}", output)
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use std::vec;

    use super::*;

    fn example_grid_u32() -> Grid<u32> {
        let grid = Grid::new(10, 10, (1..=100).collect());

        println!("Grid<u32>: ");
        println!("{}", grid);

        grid
    }

    fn example_grid_string() -> Grid<String> {
        let grid = Grid::new(
            5,
            2,
            vec!["a", "aa", "aa", "aa", "a", "aaaa", "aa", "aaaaa", "aa", "a"]
                .into_iter()
                .map(|s| s.to_owned())
                .collect(),
        );

        println!("Grid<String>: ");
        println!("{}", grid);

        grid
    }

    fn small_example_grid() -> Grid<char> {
        let grid = Grid::new(2, 3, vec!['a', 'b', 'c', 'd', 'e', 'f']);

        println!("Grid<char>: ");
        println!("{}", grid);

        grid
    }

    #[test]
    fn index_test() {
        let grid = example_grid_u32();

        assert_eq!(grid.get((5, 2)).unwrap(), &26);

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
        let grid = example_grid_u32();

        let actual_items_in_row: Vec<u32> = grid.row_iter(2).copied().collect();

        assert_eq!(
            actual_items_in_row,
            vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        );
    }

    #[test]
    fn col_iter_test() {
        let grid = example_grid_u32();

        let actual_items_in_col: Vec<u32> = grid.column_iter(2).copied().collect();

        assert_eq!(
            actual_items_in_col,
            vec![3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
        );
    }

    #[test]
    fn new_default_test() {
        let grid: Grid<u32> = Grid::new_default(10, 1);

        assert_eq!(grid.data(), &vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]);
    }

    #[test]
    fn add_row_test() {
        let mut grid = example_grid_u32();

        let items_in_row_1: Vec<u32> = grid.row_iter(1).copied().collect();

        assert_eq!(items_in_row_1, vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
        assert_eq!(grid.height(), 10);

        grid.add_row(1, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(grid.height(), 11);
    }

    #[test]
    fn remove_row_test() {
        let mut grid = example_grid_string();
        let items_in_row_1: Vec<_> = grid.row_iter(1).cloned().collect();

        assert_eq!(items_in_row_1, vec!["aaaa", "aa", "aaaaa", "aa", "a"]);
        assert_eq!(grid.height(), 2);

        grid.remove_row(1);
        assert_eq!(grid.height(), 1);
    }

    #[test]
    fn add_column_test() {
        let mut grid = example_grid_u32();

        let items_in_column_1: Vec<u32> = grid.column_iter(1).copied().collect();

        assert_eq!(
            items_in_column_1,
            vec![2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
        );
        assert_eq!(grid.width(), 10);

        grid.add_column(1, vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);

        let items_in_column_1: Vec<u32> = grid.column_iter(1).copied().collect();

        assert_eq!(items_in_column_1, vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
        assert_eq!(grid.width(), 11);
    }

    #[test]
    fn remove_column_test() {
        let mut grid = example_grid_string();
        let items_in_column_1: Vec<_> = grid.column_iter(1).cloned().collect();

        assert_eq!(items_in_column_1, vec!["aa", "aa"]);
        assert_eq!(grid.width(), 5);

        grid.remove_column(1);
        let items_in_column_1: Vec<_> = grid.column_iter(1).cloned().collect();
        assert_eq!(items_in_column_1, vec!["aa", "aaaaa"]);
        assert_eq!(grid.width(), 4);
    }

    #[test]
    fn rotate_cw_test() {
        let grid = small_example_grid();

        let rotated = grid.rotate_cw();

        assert_eq!(rotated, Grid::new(3, 2, vec!['e', 'c', 'a', 'f', 'd', 'b']));
    }

    #[test]
    fn rotate_ccw_test() {
        let grid = small_example_grid();

        let rotated = grid.rotate_ccw();

        assert_eq!(rotated, Grid::new(3, 2, vec!['b', 'd', 'f', 'a', 'c', 'e']));
    }

    #[test]
    fn flip_horizontally_test() {
        let grid = small_example_grid();

        let flipped = grid.flip_horizontally();

        assert_eq!(flipped, Grid::new(2, 3, vec!['b', 'a', 'd', 'c', 'f', 'e']));
    }

    #[test]
    fn flip_vertically_test() {
        let grid = small_example_grid();

        let flipped = grid.flip_vertically();

        assert_eq!(flipped, Grid::new(2, 3, vec!['e', 'f', 'c', 'd', 'a', 'b']));
    }
}
