//! A simple and small library for representing two-dimensional grids.

mod index;
#[cfg(feature = "linalg")]
pub mod linalg;
pub(crate) mod utils;

pub use index::GridIndex;
use index::LinearIndexError;
use std::{
    collections::HashMap,
    fmt::Display,
    ops::{Index, IndexMut},
};
use utils::*;

/// A two-dimensional array, indexed with x-and-y-coordinates (columns and rows).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Grid<T> {
    /// The width of the grid (number of columns).
    pub(crate) width: usize,
    /// The height of the grid (number of rows).
    pub(crate) height: usize,
    /// The data of the grid, stored in a linear array of `width * height` length.
    data: Vec<T>,
}

impl<T> Grid<T> {
    /// Construct a new Grid.
    ///
    /// ## Panics
    /// * If `width * height != data.len()`
    /// * If one of (but not both) `width` and `height` is zero.
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// // construct a 2x3 (width x height) grid of chars
    /// let grid = Grid::new(2, 3, "abcdef".chars().collect());
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a b
    /// // c d
    /// // e f
    /// ```
    pub fn new(width: usize, height: usize, data: Vec<T>) -> Self {
        panic_if_width_times_height_is_not_equal_to_data_len(width, height, data.len());
        panic_if_width_xor_height_is_zero(width, height);

        Self {
            width,
            height,
            data,
        }
    }

    /// Create a grid filled with default values.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let grid: Grid<bool> = Grid::new_default(2, 2);
    /// assert_eq!(grid, Grid::new(2, 2, vec![false, false, false, false]));
    /// ```
    pub fn new_default(width: usize, height: usize) -> Grid<T>
    where
        T: Default,
    {
        let mut data = Vec::with_capacity(width * height);
        for _ in 0..data.capacity() {
            data.push(T::default());
        }
        Self::new(width, height, data)
    }

    /// Construct a `Grid` from another, by converting each element.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let u32_grid: Grid<u32> = Grid::new(2, 2, vec![1, 2, 3, 4]);
    /// let f64_grid: Grid<f64> = Grid::from_grid(u32_grid);
    /// assert_eq!(f64_grid, Grid::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]));
    /// ```
    pub fn from_grid<U>(other: Grid<U>) -> Grid<T>
    where
        T: From<U>,
    {
        let (width, height) = other.dimensions();
        let mut new_data = Vec::with_capacity(other.area());
        for item in other.take_data() {
            new_data.push(T::from(item));
        }

        Grid::new(width, height, new_data)
    }

    /// Returns the width (number of columns) of the `Grid`.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height (number of rows) of the `Grid`.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns `true` if the grid contains some element equal to `value`.
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, "abcd".chars().collect());
    /// assert!(grid.contains(&'a'));
    /// assert!(!grid.contains(&'e'));
    /// ```
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.cell_iter().any(|element| element == value)
    }

    /// Consumes the `Grid`, creating a new one from a subset of the original.
    ///
    /// ## Arguments
    /// * `column_start` - Left bound for the subgrid.
    /// * `row_start` - Upper bound for the subgrid.
    /// * `width` - Number of columns in the subgrid.
    /// * `height` - Number of rows in the subgrid.
    ///
    /// ## Panics
    /// * If `width` or `height` (but not both) are 0. If both are 0, the resulting subgrid will be an empty (0 by 0) `Grid`.
    /// * If `column_start` is out of bounds.
    /// * If `row_start` is out of bounds.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let original: Grid<u32> = Grid::new(3, 3, (1..=9).collect());
    /// let subgrid = original.subgrid(1, 0, 2, 2);
    /// assert_eq!(subgrid, Grid::new(2, 2, vec![2, 3, 5, 6]));
    /// ```
    pub fn subgrid(
        self,
        column_start: usize,
        row_start: usize,
        width: usize,
        height: usize,
    ) -> Grid<T> {
        panic_if_width_xor_height_is_zero(width, height);
        panic_if_column_out_of_bounds(&self, column_start);
        panic_if_row_out_of_bounds(&self, row_start);

        let column_end = column_start + width;
        let row_end = row_start + height;

        let is_within_bounds = |idx: GridIndex| {
            idx.column() >= column_start
                && idx.column() < column_end
                && idx.row() >= row_start
                && idx.row() < row_end
        };

        let indices = self.indices();
        let mut data = self.data;

        for idx in indices.rev() {
            if !is_within_bounds(idx) {
                let linear_idx = idx.to_linear_idx_in(self.width);
                data.remove(linear_idx);
            }
        }

        Grid::new(width, height, data)
    }

    /// Returns a tuple containing the (width, height) of the grid.
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    /// assert_eq!(grid.dimensions(), (2, 3));
    /// ```
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Set new dimensions for the `Grid`.
    ///
    /// ## Panics
    /// * If the new width and height don't match the current length of the underlying vector.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut grid: Grid<u32> = Grid::new(6, 6, (0..36).collect());
    /// assert_eq!(grid.dimensions(), (6, 6));
    ///
    /// // 6*6 = 2*18, so this is fine
    /// grid.set_dimensions(2, 18);
    /// assert_eq!(grid.dimensions(), (2, 18));
    ///
    /// // but this would panic
    /// // grid.set_dimensions(2, 17);
    /// ```
    pub fn set_dimensions(&mut self, width: usize, height: usize) {
        panic_if_width_times_height_is_not_equal_to_data_len(width, height, self.data.len());

        self.width = width;
        self.height = height;
    }

    /// Checks if the Grid is square (number of columns and rows is equal).
    ///
    /// ## Note
    /// An empty Grid is not square (even though columns and rows is 0).
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, vec![1, 2, 3, 4]);
    /// assert!(grid.is_square());
    /// let grid = Grid::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    /// assert!(!grid.is_square());
    /// ```
    pub fn is_square(&self) -> bool {
        !self.is_empty() && self.width == self.height
    }

    fn is_empty(&self) -> bool {
        let ans = self.width == 0 || self.height == 0;
        if ans {
            debug_assert!(self.width == 0);
            debug_assert!(self.height == 0);
        }
        ans
    }

    /// Returns the area (number of columns * number of rows) of the grid.
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 3, vec![2, 4, 8, 16, 32, 64]);
    /// assert_eq!(grid.area(), 6);
    /// ```
    pub fn area(&self) -> usize {
        self.width * self.height
    }

    /// Attempts to get a reference to the element at `idx`.
    ///
    /// Returns `None` if `idx` is out of bounds.
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 3, vec![2, 4, 8, 16, 32, 64]);
    /// assert_eq!(grid.get((1, 1)), Some(&16));
    /// assert!(grid.get((10, 15)).is_none());
    /// ```
    pub fn get<I>(&self, idx: I) -> Option<&T>
    where
        GridIndex: From<I>,
    {
        let index: usize = self.linear_idx(GridIndex::from(idx)).ok()?;

        Some(&self.data[index])
    }

    /// Attempts to get a mutable reference to the element at `idx`
    ///
    /// Returns `None` if `idx` is out of bounds.
    pub fn get_mut<I>(&mut self, idx: I) -> Option<&mut T>
    where
        GridIndex: From<I>,
    {
        let index: usize = self.linear_idx(GridIndex::from(idx)).ok()?;

        Some(&mut self.data[index])
    }

    /// Return an iterator over the cells in the grid.
    ///
    /// Iterates from left to right (starting with row 0, then row 1 etc.).
    pub fn cell_iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.data.iter()
    }

    /// Return an iterator over the columns in the row with index `row`.
    ///
    /// ## Panics
    /// * If `row >= self.height`
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(10, 10, (1..=100).collect());
    /// let items_in_row_2: Vec<u32> = grid.row_iter(2).cloned().collect();
    /// assert_eq!(items_in_row_2, vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]);
    /// ```
    pub fn row_iter(&self, row: usize) -> impl DoubleEndedIterator<Item = &T> {
        panic_if_row_out_of_bounds(self, row);

        (0..self.width).map(move |column| &self[(column, row)])
    }

    /// Return an iterator over the rows in the column with index `column`.
    ///
    /// ## Panics
    /// * If `column >= self.width`
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(10, 10, (1..=100).collect());
    /// let items_in_column_2: Vec<u32> = grid.column_iter(2).cloned().collect();
    /// assert_eq!(items_in_column_2, vec![3, 13, 23, 33, 43, 53, 63, 73, 83, 93]);
    /// ```
    pub fn column_iter(&self, column: usize) -> impl DoubleEndedIterator<Item = &T> {
        panic_if_column_out_of_bounds(self, column);
        (0..self.height).map(move |row| &self[(column, row)])
    }

    /// Insert a row at index `row`, shifting all other rows downward (row `n` becomes row `n+1` and so on).
    ///
    /// ## Panics
    /// * If `row_contents.is_empty()`
    /// * If `row_contents.len() != self.width`
    /// * If `row > self.height` (note that `row == self.height` is allowed, to add a row at the bottom of the `Grid`)
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.insert_row(1, "xx".chars().collect());
    /// assert_eq!(grid, Grid::new(2, 3, "abxxcd".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a b
    /// // x x
    /// // c d
    /// ```
    pub fn insert_row(&mut self, row: usize, row_contents: Vec<T>) {
        panic_if_row_is_empty(&row_contents);

        if self.is_empty() && row == 0 {
            // special case, if the grid is empty, we can insert a row of any width
            self.width = row_contents.len();
            self.height = 1;
            self.data = row_contents;
            return;
        }

        panic_if_row_length_is_not_equal_to_width(self, row_contents.len());

        if row > self.height {
            // for example, if the height of the grid is 1,
            // we still want to support adding a column at the bottom
            panic!(
                "row insertion index (is {}) should be <= height (is {})",
                row, self.height
            );
        }

        let start_idx = GridIndex::new(0, row).to_linear_idx_in(self.width);

        for (elem, idx) in row_contents.into_iter().zip(start_idx..) {
            self.data.insert(idx, elem);
        }

        self.height += 1;
    }

    /// Add a row to the bottom of the `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.push_row(vec!['x', 'x']);
    /// assert_eq!(grid, Grid::new(2, 3, "abcdxx".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints
    /// // a b
    /// // c d
    /// // x x
    /// ```
    pub fn push_row(&mut self, row_contents: Vec<T>) {
        self.insert_row(self.height(), row_contents);
    }

    /// Replace the contents in a row.
    ///
    /// Returns the old elements of the row.
    ///
    /// ## Panics
    /// * If `row >= self.height`
    /// * If `data.len() != self.width`
    pub fn replace_row(&mut self, row: usize, data: Vec<T>) -> Vec<T> {
        panic_if_row_out_of_bounds(self, row);
        panic_if_row_length_is_not_equal_to_width(self, data.len());

        let mut old = Vec::with_capacity(self.width);
        for (column, elem) in data.into_iter().enumerate() {
            let old_value = self.replace_cell((column, row), elem);
            old.push(old_value);
        }
        old
    }

    /// Remove row at `row`, shifting all rows with higher indices "upward" (row `n` becomes row `n-1`).
    ///
    /// Returns the row that was removed.
    ///
    /// ## Panics
    /// * If `row >= self.height`
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.remove_row(1);
    /// assert_eq!(grid, Grid::new(2, 1, "ab".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a b
    /// ```
    pub fn remove_row(&mut self, row: usize) -> Vec<T> {
        panic_if_row_out_of_bounds(self, row);

        let start_idx = self.linear_idx(GridIndex::new(0, row)).unwrap();

        let r: Vec<T> = self.data.drain(start_idx..start_idx + self.width).collect();
        self.height -= 1;

        if self.height == 0 {
            //  no rows remain, so the grid is empty
            self.width = 0;
        }
        r
    }

    /// Remove the bottom row, returning it (if it exists).
    ///
    /// Returns `None` if the height of the `Grid` is zero.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// let bottom_row = grid.pop_row();
    /// assert_eq!(bottom_row, Some(vec!['c', 'd']));
    /// assert_eq!(grid, Grid::new(2, 1, "ab".chars().collect()));
    /// ```
    pub fn pop_row(&mut self) -> Option<Vec<T>> {
        if self.height() == 0 {
            None
        } else {
            Some(self.remove_row(self.height() - 1))
        }
    }

    /// Swap two rows in the grid.
    ///
    /// ## Panics
    /// * If either of the row indices are out of bounds.
    pub fn swap_rows(&mut self, row1: usize, row2: usize) {
        panic_if_row_out_of_bounds(self, row1);
        panic_if_row_out_of_bounds(self, row2);

        if row1 != row2 {
            for column in self.columns() {
                let linear_idx1 = self.linear_idx(GridIndex::new(column, row1)).unwrap();
                let linear_idx2 = self.linear_idx(GridIndex::new(column, row2)).unwrap();
                self.data.swap(linear_idx1, linear_idx2);
            }
        }
    }

    /// Insert a column at index `column`, shifting all other columns to the right (column `n` becomes column `n+1` and so on).
    ///
    /// ## Panics
    /// * If `column_contents.is_empty()`
    /// * If `column_contents.len() != self.height`
    /// * If `column > self.width` (note that `column == self.width` is allowed, to add a column at the right of the `Grid`)
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.insert_column(1, "xx".chars().collect());
    /// assert_eq!(grid, Grid::new(3, 2, "axbcxd".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a x b
    /// // c x d
    /// ```
    pub fn insert_column(&mut self, column: usize, column_contents: Vec<T>) {
        panic_if_column_is_empty(&column_contents);

        if self.is_empty() && column == 0 {
            // special case, if the grid is empty, we can insert a column of any height
            self.height = column_contents.len();
            self.width = 1;
            self.data = column_contents;
            return;
        }

        panic_if_column_length_is_not_equal_to_height(self, column_contents.len());

        if column > self.width {
            // for example, if the width of the grid is 1,
            // we still want to support adding a column at the furthest right
            panic!(
                "column insertion index (is {}) should be <= width (is {})",
                column, self.width
            );
        }

        let indices: Vec<usize> = (0..column_contents.len())
            .map(|row| GridIndex::new(column, row).to_linear_idx_in(self.width + 1))
            .collect();

        for (elem, idx) in column_contents.into_iter().zip(indices.into_iter()) {
            self.data.insert(idx, elem);
        }

        self.width += 1;
    }

    /// Add a column to the right of the `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.push_column(vec!['x', 'x']);
    /// assert_eq!(grid, Grid::new(3, 2, "abxcdx".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints
    /// // a b x
    /// // c d x
    /// ```
    pub fn push_column(&mut self, column_contents: Vec<T>) {
        self.insert_column(self.width(), column_contents);
    }

    /// Replace the contents in a column.
    ///
    /// Returns the old elements of the column.
    ///
    /// ## Panics
    /// * If `column >= self.width`
    /// * If `data.len() != self.height`
    pub fn replace_column(&mut self, column: usize, data: Vec<T>) -> Vec<T> {
        panic_if_column_out_of_bounds(self, column);
        panic_if_column_length_is_not_equal_to_height(self, data.len());

        let mut old = Vec::with_capacity(self.height);
        for (row, elem) in data.into_iter().enumerate() {
            let old_value = self.replace_cell((column, row), elem);
            old.push(old_value);
        }

        old
    }

    /// Swap two columns in the grid.
    ///
    /// ## Panics
    /// * If either of the column indices are out of bounds.
    pub fn swap_columns(&mut self, column1: usize, column2: usize) {
        panic_if_column_out_of_bounds(self, column1);
        panic_if_column_out_of_bounds(self, column2);

        if column1 != column2 {
            for row in self.rows() {
                let linear_idx1 = self.linear_idx(GridIndex::new(column1, row)).unwrap();
                let linear_idx2 = self.linear_idx(GridIndex::new(column2, row)).unwrap();
                self.data.swap(linear_idx1, linear_idx2);
            }
        }
    }

    /// Remove column at `column`, shifting all columns with higher indices "left" (column `n` becomes column `n-1`).
    ///
    /// Returns the column that was removed.
    ///
    /// ## Panics
    /// * If `column >= self.width`
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.remove_column(1);
    /// assert_eq!(grid, Grid::new(1, 2, "ac".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a
    /// // c
    /// ```
    pub fn remove_column(&mut self, column: usize) -> Vec<T> {
        panic_if_column_out_of_bounds(self, column);

        let indices: Vec<usize> = self
            .rows()
            .map(|row| self.linear_idx(GridIndex::new(column, row)).unwrap())
            .collect();

        let mut c = Vec::with_capacity(self.height);

        for idx in indices.into_iter().rev() {
            let elem = self.data.remove(idx);
            c.insert(0, elem);
        }

        self.width -= 1;
        if self.width == 0 {
            //  no columns remain, so the grid is empty
            self.height = 0;
        }

        c
    }

    /// Remove the rightmost column, returning it (if it exists).
    ///
    /// Returns `None` if the width of the `Grid` is zero.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// let rightmost_column = grid.pop_column();
    /// assert_eq!(rightmost_column, Some(vec!['b', 'd']));
    /// assert_eq!(grid, Grid::new(1, 2, "ac".chars().collect()));
    /// ```
    pub fn pop_column(&mut self) -> Option<Vec<T>> {
        if self.width() == 0 {
            None
        } else {
            Some(self.remove_column(self.width() - 1))
        }
    }

    /// Swap the values in two cells in the grid.
    ///
    /// ## Panics
    /// * If either index is out of bounds.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    /// grid.swap_cells((1, 1), (0, 2));
    /// assert_eq!(grid, Grid::new(2, 3, vec![1, 2, 3, 5, 4, 6]));
    /// ```
    pub fn swap_cells<I>(&mut self, a: I, b: I)
    where
        GridIndex: From<I>,
    {
        let a_idx = GridIndex::from(a);
        let b_idx = GridIndex::from(b);

        panic_if_index_out_of_bounds(self, a_idx);
        panic_if_index_out_of_bounds(self, b_idx);

        let a_linear = self.linear_idx(a_idx).unwrap();
        let b_linear = self.linear_idx(b_idx).unwrap();
        self.data.swap(a_linear, b_linear);
    }

    /// Replace the contents in a cell.
    ///
    /// Returns the old element of the cell.
    ///
    /// ## Panics
    /// * If `idx` is out of bounds.
    pub fn replace_cell<I>(&mut self, idx: I, elem: T) -> T
    where
        GridIndex: From<I>,
    {
        let idx = GridIndex::from(idx);
        panic_if_index_out_of_bounds(self, idx);
        let linear = self.linear_idx_unchecked(idx);
        std::mem::replace(&mut self.data[linear], elem)
    }

    /// Rotate the grid counter-clockwise 90 degrees.
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a b
    /// // c d
    ///
    /// grid.rotate_ccw();
    /// assert_eq!(grid, Grid::new(2, 2, "bdac".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // b d
    /// // a c
    /// ```
    pub fn rotate_ccw(&mut self) {
        if self.is_empty() {
            return;
        }

        let mut target_index = HashMap::new();
        let mut current_target = 0;
        for column in self.columns().rev() {
            for row in self.rows() {
                let from = self.linear_idx(GridIndex::new(column, row)).unwrap();
                target_index.insert(from, current_target);
                current_target += 1;
            }
        }

        self.transform(target_index);

        std::mem::swap(&mut self.width, &mut self.height);
    }

    /// Rotate the grid clockwise 90 degrees.
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a b
    /// // c d
    ///
    /// grid.rotate_cw();
    /// assert_eq!(grid, Grid::new(2, 2, "cadb".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // c a
    /// // d b
    /// ```
    pub fn rotate_cw(&mut self) {
        if self.is_empty() {
            return;
        }

        let mut target_index = HashMap::new();
        let mut current_target = 0;
        for column in self.columns() {
            for row in self.rows().rev() {
                let from = self.linear_idx(GridIndex::new(column, row)).unwrap();
                target_index.insert(from, current_target);
                current_target += 1;
            }
        }

        self.transform(target_index);

        std::mem::swap(&mut self.width, &mut self.height);
    }

    /// Flip the grid horizontally, so that the first column becomes the last.
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a b
    /// // c d
    ///
    /// grid.flip_horizontally();
    /// assert_eq!(grid, Grid::new(2, 2, "badc".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // b a
    /// // d c
    /// ```
    pub fn flip_horizontally(&mut self) {
        if self.is_empty() {
            return;
        }

        let mut target_index = HashMap::new();
        let mut current_target = 0;
        for row in self.rows() {
            for column in self.columns().rev() {
                let from = self.linear_idx(GridIndex::new(column, row)).unwrap();
                target_index.insert(from, current_target);
                current_target += 1;
            }
        }

        self.transform(target_index);
    }

    /// Flip the grid vertically, so that the first row becomes the last.
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a b
    /// // c d
    ///
    /// grid.flip_vertically();
    /// assert_eq!(grid, Grid::new(2, 2, "cdab".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // c d
    /// // a b
    /// ```
    pub fn flip_vertically(&mut self) {
        if self.is_empty() {
            return;
        }

        let mut target_index = HashMap::new();
        let mut current_target = 0;
        for row in self.rows().rev() {
            for column in self.columns() {
                let from = self.linear_idx(GridIndex::new(column, row)).unwrap();
                target_index.insert(from, current_target);
                current_target += 1;
            }
        }

        self.transform(target_index);
    }

    /// Transpose the grid along the diagonal, so that cells at index (x, y) end up at index (y, x).
    ///
    /// ## Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 3, "abcdef".chars().collect());
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a b
    /// // c d
    /// // e f
    ///
    /// grid.transpose();
    /// assert_eq!(grid, Grid::new(3, 2, "acebdf".chars().collect()));
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// // a c e
    /// // b d f
    /// ```
    pub fn transpose(&mut self) {
        if self.is_empty() {
            return;
        }

        let mut target_index = HashMap::new();
        let mut current_target = 0;
        for column in self.columns() {
            for row in self.rows() {
                let idx = GridIndex::new(column, row);
                let from = self.linear_idx(idx).unwrap();
                target_index.insert(from, current_target);
                current_target += 1;
            }
        }

        self.transform(target_index);

        std::mem::swap(&mut self.width, &mut self.height);
    }

    /// Transforms the Grid, moving the contents of cells to new indices based on a hashmap.
    fn transform(&mut self, mut target_index: HashMap<usize, usize>) {
        while !target_index.is_empty() {
            let current = *target_index.keys().next().unwrap();
            let mut target = target_index.remove(&current).unwrap();

            loop {
                // swap current with its target until a cycle has been reached
                self.data.swap(current, target);
                match target_index.remove(&target) {
                    Some(t) => target = t,
                    None => {
                        break;
                    }
                }
            }
        }
    }

    /// Convert a GridIndex into an index in the internal data of the Grid.
    fn linear_idx(&self, idx: GridIndex) -> Result<usize, LinearIndexError> {
        if idx.row() >= self.height {
            Err(LinearIndexError::RowTooHigh)
        } else if idx.column() >= self.width {
            Err(LinearIndexError::ColumnTooHigh)
        } else {
            Ok(idx.to_linear_idx_in(self.width))
        }
    }

    /// Same as `linear_idx`, but panics when `idx` is out of bounds.
    fn linear_idx_unchecked(&self, idx: GridIndex) -> usize {
        panic_if_index_out_of_bounds(self, idx);
        idx.to_linear_idx_in(self.width)
    }

    /// Return an iterator over the row indices in this grid.
    ///
    /// Allows you to write `for row in grid.rows()` instead of `for row in 0..grid.height()`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let grid: Grid<u32> = Grid::new_default(3, 5);
    /// let rows: Vec<usize> = grid.rows().collect();
    /// assert_eq!(rows, vec![0, 1, 2, 3, 4]);
    /// ```
    pub fn rows(&self) -> impl DoubleEndedIterator<Item = usize> {
        0..self.height
    }

    /// Return an iterator over the column indices in this grid.
    ///
    /// Allows you to write `for column in grid.columns()` instead of `for column in 0..grid.width()`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let grid: Grid<u32> = Grid::new_default(2, 5);
    /// let rows: Vec<usize> = grid.columns().collect();
    /// assert_eq!(rows, vec![0, 1]);
    /// ```
    pub fn columns(&self) -> impl DoubleEndedIterator<Item = usize> {
        0..self.width
    }

    /// Searches for an element in the `Grid` matching a predicate, returning its index.
    ///
    /// Iterates from left to right (looks through row 0 followed by row 1 etc.).
    ///
    /// Returns the index of the first element that matches the predicate.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let grid = Grid::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    /// let position_of_4 = grid.position(|&e| e == 4);
    /// assert_eq!(position_of_4, Some(GridIndex::new(1, 1)));
    /// ```
    pub fn position<P>(&self, predicate: P) -> Option<GridIndex>
    where
        P: Fn(&T) -> bool,
    {
        for idx in self.indices() {
            let elem = &self[idx];
            if predicate(elem) {
                return Some(idx);
            }
        }
        None
    }

    /// Return an iterator over the cell indices in this grid.
    ///
    /// Iterates from left to right (through row 0 followed by row 1 etc.).
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_three: Grid<u32> = Grid::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    /// let indices: Vec<GridIndex> = two_by_three.indices().collect();
    /// assert_eq!(indices, vec![GridIndex::new(0, 0), GridIndex::new(1, 0), GridIndex::new(0, 1), GridIndex::new(1, 1), GridIndex::new(0, 2), GridIndex::new(1, 2)]);
    /// ```
    pub fn indices(&self) -> impl DoubleEndedIterator<Item = GridIndex> {
        let height = self.height;
        let width = self.width;
        (0..height).flat_map(move |row| (0..width).map(move |column| GridIndex::new(column, row)))
    }

    /// Return an iterator over the cells in the grid, together with their indices.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let grid = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(grid.cells_with_indices_iter().collect::<Vec<_>>(), vec![(GridIndex::new(0, 0), &'a'), (GridIndex::new(1, 0), &'b'), (GridIndex::new(0, 1), &'c'), (GridIndex::new(1, 1), &'d')]);
    /// ```
    pub fn cells_with_indices_iter(&self) -> impl DoubleEndedIterator<Item = (GridIndex, &T)> {
        self.indices().map(move |idx| (idx, &self[idx]))
    }

    /// Returns `true` if `idx` is within the bounds of this `Grid`, `false` otherwise.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert!(two_by_two.contains_index(GridIndex::new(1, 1)));
    /// assert!(!two_by_two.contains_index(GridIndex::new(2, 1)));
    /// ```
    pub fn contains_index(&self, idx: GridIndex) -> bool {
        idx.row() < self.height() && idx.column() < self.width()
    }

    /// Returns an iterator over the indices of the cardinal and ordinal neighbors of the cell at `idx`.
    ///
    /// Returns the neighbors in clockwise order: `[up, up_right, right, down_right, down, down_left, left, up_left]`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let three_by_three = Grid::new(3, 3, "abcdefghi".chars().collect());
    /// // a b c
    /// // d e f
    /// // g h i
    /// let neighbors: Vec<_> = three_by_three.neighbor_indices_of((1, 1)).collect();
    /// assert_eq!(neighbors, vec![
    ///     (1, 0).into(), // up
    ///     (2, 0).into(), // up_right
    ///     (2, 1).into(), // right
    ///     (2, 2).into(), // down_right
    ///     (1, 2).into(), // down
    ///     (0, 2).into(), // down_left
    ///     (0, 1).into(), // left
    ///     (0, 0).into(), // up_left
    /// ]);
    /// ```
    pub fn neighbor_indices_of<I>(&'_ self, idx: I) -> impl Iterator<Item = GridIndex> + '_
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        idx.neighbors().filter(move |i| self.contains_index(*i))
    }

    /// Returns an iterator over the contents of the cardinal and ordinal neighbors of the cell at `idx`.
    ///
    /// Returns the neighbors in clockwise order: `[up, up_right, right, down_right, down, down_left, left, up_left]`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let three_by_three = Grid::new(3, 3, "abcdefghi".chars().collect());
    /// // a b c
    /// // d e f
    /// // g h i
    /// let neighbors: Vec<_> = three_by_three.neighbor_cells_of((1, 1)).collect();
    /// assert_eq!(neighbors, vec![
    ///     &'b', // up
    ///     &'c', // up_right
    ///     &'f', // right
    ///     &'i', // down_right
    ///     &'h', // down
    ///     &'g', // down_left
    ///     &'d', // left
    ///     &'a', // up_left
    /// ]);
    /// ```
    pub fn neighbor_cells_of<I>(&self, idx: I) -> impl Iterator<Item = &T>
    where
        I: Into<GridIndex>,
    {
        self.neighbor_indices_of(idx).map(move |i| &self[i])
    }

    /// Returns an iterator over the indices of the cardinal neighbors of the cell at `idx`.
    ///
    /// Returns the neighbors in clockwise order: `[up, right, down, left]`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// let neighbors: Vec<_> = two_by_two.cardinal_neighbor_indices_of((1, 1)).collect();
    /// assert_eq!(
    ///     neighbors,
    ///     vec![GridIndex::new(1, 0), GridIndex::new(0, 1)]
    /// );
    /// ```
    pub fn cardinal_neighbor_indices_of<I>(&'_ self, idx: I) -> impl Iterator<Item = GridIndex> + '_
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        idx.cardinal_neighbors()
            .filter(move |i| self.contains_index(*i))
    }

    /// Returns an iterator over the contents of the cardinal neighbors of the cell at `idx`.
    ///
    /// Returns the neighbors in clockwise order: `[up, right, down, left]`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// let neighbors: Vec<_> = two_by_two.cardinal_neighbor_cells_of((1, 1)).collect();
    /// assert_eq!(neighbors, vec![&'b', &'c']);
    /// ```
    pub fn cardinal_neighbor_cells_of<I>(&self, idx: I) -> impl Iterator<Item = &T>
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        self.cardinal_neighbor_indices_of(idx)
            .map(move |i| &self[i])
    }

    /// Returns the `GridIndex` above `idx`, if it exists.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.up_index((1, 1)), Some(GridIndex::new(1, 0)));
    /// assert_eq!(two_by_two.up_index((1, 0)), None);
    /// ```
    pub fn up_index<I>(&self, idx: I) -> Option<GridIndex>
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        if let Some(up) = idx.up() {
            if self.contains_index(up) {
                return Some(up);
            }
        }
        None
    }

    /// Returns a reference to the contents of the cell above `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.up_cell((1, 1)), Some(&'b'));
    /// assert_eq!(two_by_two.up_cell((1, 0)), None);
    /// ```
    pub fn up_cell<I>(&self, idx: I) -> Option<&T>
    where
        I: Into<GridIndex>,
    {
        self.up_index(idx).map(|i| &self[i])
    }

    /// Returns the `GridIndex` above and to the right of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.up_right_index((0, 1)), Some(GridIndex::new(1, 0)));
    /// assert_eq!(two_by_two.up_right_index((1, 0)), None);
    /// ```
    pub fn up_right_index<I>(&self, idx: I) -> Option<GridIndex>
    where
        I: Into<GridIndex>,
    {
        self.up_index(idx).and_then(|up| self.right_index(up))
    }

    /// Returns a reference to the contents of the cell above and to the right of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.up_cell((1, 1)), Some(&'b'));
    /// assert_eq!(two_by_two.up_cell((1, 0)), None);
    /// ```
    pub fn up_right_cell<I>(&self, idx: I) -> Option<&T>
    where
        I: Into<GridIndex>,
    {
        self.up_right_index(idx).map(|i| &self[i])
    }

    /// Returns the `GridIndex` to the right of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.right_index((0, 0)), Some(GridIndex::new(1, 0)));
    /// assert_eq!(two_by_two.right_index((1, 0)), None);
    /// ```
    pub fn right_index<I>(&self, idx: I) -> Option<GridIndex>
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        let right = idx.right()?;
        if self.contains_index(right) {
            Some(right)
        } else {
            None
        }
    }

    /// Returns a reference to the contents of the cell to the right of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.right_cell((0, 1)), Some(&'d'));
    /// assert_eq!(two_by_two.right_cell((1, 0)), None);
    /// ```
    pub fn right_cell<I>(&self, idx: I) -> Option<&T>
    where
        I: Into<GridIndex>,
    {
        self.right_index(idx).map(|i| &self[i])
    }

    /// Returns the `GridIndex` below and to the right of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.down_right_index((0, 0)), Some(GridIndex::new(1,1)));
    /// assert_eq!(two_by_two.down_right_index((1, 0)), None);
    /// ```
    pub fn down_right_index<I>(&self, idx: I) -> Option<GridIndex>
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        let down_right = idx.down_right()?;
        if self.contains_index(down_right) {
            Some(down_right)
        } else {
            None
        }
    }

    /// Returns a reference to the contents of the cell below and to the right of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.down_right_cell((0, 0)), Some(&'d'));
    /// assert_eq!(two_by_two.down_right_cell((1, 0)), None);
    /// ```
    pub fn down_right_cell<I>(&self, idx: I) -> Option<&T>
    where
        I: Into<GridIndex>,
    {
        self.down_right_index(idx).map(|i| &self[i])
    }

    /// Returns the `GridIndex` below `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.down_index((0, 0)), Some(GridIndex::new(0, 1)));
    /// assert_eq!(two_by_two.down_index((0, 1)), None);
    /// ```
    pub fn down_index<I>(&self, idx: I) -> Option<GridIndex>
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        let down = idx.down()?;
        if self.contains_index(down) {
            Some(down)
        } else {
            None
        }
    }

    /// Returns a reference to the contents of the cell below `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.down_cell((0, 0)), Some(&'c'));
    /// assert_eq!(two_by_two.down_cell((0, 1)), None);
    /// ```
    pub fn down_cell<I>(&self, idx: I) -> Option<&T>
    where
        I: Into<GridIndex>,
    {
        self.down_index(idx).map(|i| &self[i])
    }

    /// Returns the `GridIndex` below and to the left of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.down_left_index((1, 0)), Some(GridIndex::new(0,1)));
    /// assert_eq!(two_by_two.down_left_index((0, 0)), None);
    /// ```
    pub fn down_left_index<I>(&self, idx: I) -> Option<GridIndex>
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        let down_left = idx.down_left()?;
        if self.contains_index(down_left) {
            Some(down_left)
        } else {
            None
        }
    }

    /// Returns a reference to the contents of the cell below and to the left of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.down_left_cell((1, 0)), Some(&'c'));
    /// assert_eq!(two_by_two.down_left_cell((0, 0)), None);
    /// ```
    pub fn down_left_cell<I>(&self, idx: I) -> Option<&T>
    where
        I: Into<GridIndex>,
    {
        self.down_left_index(idx).map(|i| &self[i])
    }

    /// Returns the `GridIndex` to the left of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.left_index((1, 0)), Some(GridIndex::new(0, 0)));
    /// assert_eq!(two_by_two.left_index((0, 0)), None);
    /// ```
    pub fn left_index<I>(&self, idx: I) -> Option<GridIndex>
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        let left = idx.left()?;
        if self.contains_index(left) {
            Some(left)
        } else {
            None
        }
    }

    /// Returns a reference to the contents of the cell to the left of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.left_cell((1, 0)), Some(&'a'));
    /// assert_eq!(two_by_two.left_cell((0, 0)), None);
    /// ```
    pub fn left_cell<I>(&self, idx: I) -> Option<&T>
    where
        I: Into<GridIndex>,
    {
        self.left_index(idx).map(|i| &self[i])
    }

    /// Returns the `GridIndex` above and to the left of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.up_left_index((1, 1)), Some(GridIndex::new(0, 0)));
    /// assert_eq!(two_by_two.up_left_index((0, 0)), None);
    /// ```
    pub fn up_left_index<I>(&self, idx: I) -> Option<GridIndex>
    where
        I: Into<GridIndex>,
    {
        let idx: GridIndex = idx.into();
        let up_left = idx.up_left()?;
        if self.contains_index(up_left) {
            Some(up_left)
        } else {
            None
        }
    }

    /// Returns a reference to the contents of the cell above and to the left of `idx`, if it exists in this `Grid`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let two_by_two = Grid::new(2, 2, "abcd".chars().collect());
    /// // a b
    /// // c d
    /// assert_eq!(two_by_two.up_left_cell((1, 1)), Some(&'a'));
    /// assert_eq!(two_by_two.up_left_cell((0, 0)), None);
    /// ```
    pub fn up_left_cell<I>(&self, idx: I) -> Option<&T>
    where
        I: Into<GridIndex>,
    {
        self.up_left_index(idx).map(|i| &self[i])
    }

    pub(crate) fn take_data(self) -> Vec<T> {
        self.data
    }
}

impl<T> Grid<T>
where
    T: Display,
{
    /// Format this `Grid` as a string. This can look weird when the `Display` output of `T` has varying length.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(10, 10, (1..=100).collect::<Vec<u32>>());
    ///
    /// println!("{}", grid.to_pretty_string());
    /// // prints:
    /// //  1   2   3   4   5   6   7   8   9  10
    /// // 11  12  13  14  15  16  17  18  19  20
    /// // 21  22  23  24  25  26  27  28  29  30
    /// // 31  32  33  34  35  36  37  38  39  40
    /// // 41  42  43  44  45  46  47  48  49  50
    /// // 51  52  53  54  55  56  57  58  59  60
    /// // 61  62  63  64  65  66  67  68  69  70
    /// // 71  72  73  74  75  76  77  78  79  80
    /// // 81  82  83  84  85  86  87  88  89  90
    /// // 91  92  93  94  95  96  97  98  99 100
    /// ```
    pub fn to_pretty_string(&self) -> String
    where
        T: Display,
    {
        let output = if let Some(max_length) = self.cell_iter().map(|c| c.to_string().len()).max() {
            let padded_string = |orig: &str| {
                let mut padding: String = " ".repeat(max_length - orig.len());
                padding.push_str(orig);
                padding
            };
            self.rows()
                .map(|r| {
                    self.columns()
                        .map(|c| padded_string(&self[(c, r)].to_string()))
                        .collect::<Vec<String>>()
                        .join(" ")
                })
                .collect::<Vec<String>>()
                .join("\n")
        } else {
            "".to_string()
        };
        output
    }
}

impl<T> IntoIterator for Grid<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Grid<T> {
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<T, I> Index<I> for Grid<T>
where
    GridIndex: From<I>,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        let index: GridIndex = GridIndex::from(index);

        let linear = self.linear_idx_unchecked(index);

        &self.data[linear]
    }
}

impl<T, I> IndexMut<I> for Grid<T>
where
    GridIndex: From<I>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index: GridIndex = GridIndex::from(index);

        let linear = self.linear_idx_unchecked(index);

        &mut self.data[linear]
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use super::*;
    use std::fmt::{Debug, Display};

    /// 1   2   3   4   5   6   7   8   9   10
    ///
    /// 11  12  13  14  15  16  17  18  19  20
    ///
    /// 21  22  23  24  25  26  27  28  29  30
    ///
    /// 31  32  33  34  35  36  37  38  39  40
    ///
    /// 41  42  43  44  45  46  47  48  49  50
    ///
    /// 51  52  53  54  55  56  57  58  59  60
    ///
    /// 61  62  63  64  65  66  67  68  69  70
    ///
    /// 71  72  73  74  75  76  77  78  79  80
    ///
    /// 81  82  83  84  85  86  87  88  89  90
    ///
    /// 91  92  93  94  95  96  97  98  99 100
    fn example_grid_u32() -> Grid<u32> {
        let grid = Grid::new(10, 10, (1..=100).collect());

        println!("Grid<u32>: ");
        println!("{}", grid.to_pretty_string());

        grid
    }

    /// a b
    ///
    /// c d
    ///
    /// e f
    fn small_example_grid() -> Grid<char> {
        let grid = Grid::new(2, 3, "abcdef".chars().collect());

        println!("Grid<char>: ");
        println!("{}", grid.to_pretty_string());

        grid
    }

    #[test]
    fn index_test() {
        let grid = example_grid_u32();

        assert_eq!(grid.get((5, 2)).unwrap(), &26);

        let mut counter = 0;
        for row in 0..grid.height {
            for col in 0..grid.width {
                counter += 1;
                assert_eq!(grid[(col, row)], counter);
            }
        }

        // this should fail
        let result = std::panic::catch_unwind(|| grid[(11, 11)]);
        assert!(result.is_err());
    }

    #[test]
    fn set_value_test() {
        let mut grid = small_example_grid();

        *grid.get_mut((0, 1)).unwrap() = 'x';

        assert_grid_equal(&grid, &Grid::new(2, 3, "abxdef".chars().collect()));

        grid[(0, 2)] = 'y';

        assert_grid_equal(&grid, &Grid::new(2, 3, "abxdyf".chars().collect()));
    }

    #[test]
    fn iter_test() {
        let grid = small_example_grid();

        for x in &grid {}
    }

    #[test]
    fn row_iter_test() {
        let grid = example_grid_u32();

        let actual_items_in_row: Vec<u32> = grid.row_iter(2).copied().collect();

        assert_eq!(
            actual_items_in_row,
            vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        );

        let actual_items_in_row_rev: Vec<u32> = grid.row_iter(2).rev().copied().collect();

        assert_eq!(
            actual_items_in_row_rev,
            vec![30, 29, 28, 27, 26, 25, 24, 23, 22, 21]
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

        assert_eq!(grid.data, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]);
    }

    #[test]
    fn insert_row_in_middle_test() {
        let mut grid = example_grid_u32();

        let items_in_row_1: Vec<u32> = grid.row_iter(1).copied().collect();

        assert_eq!(items_in_row_1, vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
        assert_eq!(grid.height, 10);

        grid.insert_row(1, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(grid.height, 11);
    }

    #[test]
    fn insert_row_at_end_test() {
        let mut grid = small_example_grid();
        let new_row: Vec<char> = "xx".chars().collect();

        grid.insert_row(3, new_row.clone());

        let items_in_bottom_row: Vec<char> = grid.row_iter(3).copied().collect();

        assert_eq!(items_in_bottom_row, new_row);

        assert_grid_equal(
            &grid,
            &Grid::new(2, 4, "abcdefxx".chars().collect::<Vec<_>>()),
        );
    }

    #[test]
    fn remove_row_test() {
        let mut grid = small_example_grid();
        let items_in_row_1: Vec<char> = grid.row_iter(1).cloned().collect();

        assert_eq!(items_in_row_1, vec!['c', 'd']);
        assert_eq!(grid.height, 3);

        let removed_row = grid.remove_row(1);
        assert_eq!(removed_row, items_in_row_1);
        assert_eq!(grid.height, 2);
    }

    #[test]
    fn remove_row_until_empty_test() {
        let mut grid = small_example_grid();

        grid.remove_row(0);
        grid.remove_row(0);
        grid.remove_row(0);

        assert_grid_equal(&grid, &Grid::new(0, 0, Vec::new()));
        assert_eq!(grid.height, 0);
        assert_eq!(grid.width, 0);
        assert!(grid.is_empty());

        // since the grid is now empty, we can add a row of any non-zero length
        grid.insert_row(0, vec!['a', 'b', 'c']);

        assert_grid_equal(&grid, &Grid::new(3, 1, vec!['a', 'b', 'c']));
    }

    #[test]
    fn insert_column_in_middle_test() {
        let mut grid = small_example_grid();

        let items_in_column_1: Vec<char> = grid.column_iter(1).copied().collect();

        assert_eq!(items_in_column_1, "bdf".chars().collect::<Vec<_>>());
        assert_eq!(grid.width, 2);

        grid.insert_column(1, "xxx".chars().collect());

        let items_in_column_1: Vec<char> = grid.column_iter(1).copied().collect();

        assert_eq!(items_in_column_1, "xxx".chars().collect::<Vec<_>>());
        assert_eq!(grid.width, 3);
    }

    #[test]
    fn insert_column_at_end_test() {
        let mut grid = small_example_grid();
        let new_column: Vec<char> = "xxx".chars().collect();
        grid.insert_column(2, new_column.clone());

        let items_in_column_2: Vec<char> = grid.column_iter(2).copied().collect();

        assert_eq!(items_in_column_2, new_column);

        assert_grid_equal(
            &grid,
            &Grid::new(3, 3, "abxcdxefx".chars().collect::<Vec<_>>()),
        );
    }

    #[test]
    fn remove_column_test() {
        let mut grid = small_example_grid();
        let items_in_column_1: Vec<_> = grid.column_iter(1).cloned().collect();

        assert_eq!(items_in_column_1, vec!['b', 'd', 'f']);
        assert_eq!(grid.width, 2);

        let removed = grid.remove_column(1);
        assert_eq!(removed, items_in_column_1);
        assert_eq!(grid.width, 1);
    }

    #[test]
    fn remove_column_until_empty_test() {
        let mut grid = small_example_grid();

        grid.remove_column(0);
        grid.remove_column(0);

        assert_grid_equal(&grid, &Grid::new(0, 0, Vec::new()));
        assert_eq!(grid.height, 0);
        assert_eq!(grid.width, 0);
        assert!(grid.is_empty());

        // since the grid is now empty, we can add a column of any non-zero length
        grid.insert_column(0, vec!['a', 'b', 'c']);

        assert_grid_equal(&grid, &Grid::new(1, 3, vec!['a', 'b', 'c']));
    }

    #[test]
    fn rotate_cw_test() {
        let mut grid = small_example_grid();

        grid.rotate_cw();

        assert_grid_equal(&grid, &Grid::new(3, 2, vec!['e', 'c', 'a', 'f', 'd', 'b']));
    }

    #[test]
    fn rotate_ccw_test() {
        let mut grid = small_example_grid();

        grid.rotate_ccw();

        assert_grid_equal(&grid, &Grid::new(3, 2, vec!['b', 'd', 'f', 'a', 'c', 'e']));
    }

    #[test]
    fn flip_horizontally_test() {
        let mut grid = small_example_grid();

        grid.flip_horizontally();

        assert_grid_equal(&grid, &Grid::new(2, 3, vec!['b', 'a', 'd', 'c', 'f', 'e']));
    }

    #[test]
    fn flip_vertically_test() {
        let mut grid = small_example_grid();

        grid.flip_vertically();

        assert_grid_equal(&grid, &Grid::new(2, 3, vec!['e', 'f', 'c', 'd', 'a', 'b']));
    }

    #[test]
    fn transpose_test() {
        let original_grid = small_example_grid();
        let mut grid = original_grid.clone();
        grid.transpose();

        assert_grid_equal(&grid, &Grid::new(3, 2, "acebdf".chars().collect()));

        grid.transpose();

        assert_grid_equal(&grid, &original_grid);
    }

    #[test]
    fn contains_test() {
        let grid = small_example_grid();

        assert!(grid.contains(&'a'));
        assert!(!grid.contains(&'g'));
    }

    #[test]
    fn is_empty_test() {
        let mut grid = small_example_grid();

        assert!(!grid.is_empty());

        grid.remove_row(0);
        assert!(!grid.is_empty());

        grid.remove_row(0);
        assert!(!grid.is_empty());

        grid.remove_row(0);
        assert!(grid.is_empty());

        grid.insert_row(0, vec!['g', 'h', 'i', 'j', 'k']);

        assert!(!grid.is_empty());
        assert_eq!(grid.width, 5);
    }

    #[test]
    fn replace_row_test() {
        let mut grid = small_example_grid();

        let items_in_row_1: Vec<char> = grid.row_iter(1).copied().collect();
        let old_row = grid.replace_row(1, vec!['x', 'x']);

        assert_eq!(old_row, items_in_row_1);
        assert_grid_equal(&grid, &Grid::new(2, 3, "abxxef".chars().collect()));
    }

    #[test]
    fn replace_column_test() {
        let mut grid = small_example_grid();

        let items_in_column_0: Vec<char> = grid.column_iter(0).copied().collect();
        let old_column = grid.replace_column(0, vec!['x', 'x', 'x']);

        assert_eq!(old_column, items_in_column_0);
        assert_grid_equal(&grid, &Grid::new(2, 3, "xbxdxf".chars().collect()));
    }

    #[test]
    fn swap_columns_test() {
        let mut grid = small_example_grid();

        grid.swap_columns(0, 1);

        assert_grid_equal(&grid, &Grid::new(2, 3, "badcfe".chars().collect()));
    }

    #[test]
    fn swap_rows_test() {
        let mut grid = small_example_grid();

        grid.swap_rows(1, 2);

        assert_grid_equal(&grid, &Grid::new(2, 3, "abefcd".chars().collect()));
    }

    #[test]
    fn swap_cells_test() {
        let mut grid = small_example_grid();
        grid.swap_cells((1, 1), (0, 2));

        assert_grid_equal(&grid, &Grid::new(2, 3, "abcedf".chars().collect()));
    }

    #[test]
    fn subgrid_test() {
        let grid = example_grid_u32();
        let subgrid = grid.subgrid(2, 1, 3, 5);
        assert_grid_equal(
            &subgrid,
            &Grid::new(
                3,
                5,
                vec![13, 14, 15, 23, 24, 25, 33, 34, 35, 43, 44, 45, 53, 54, 55],
            ),
        );
    }

    #[test]
    fn replace_cell_test() {
        let mut grid = small_example_grid();
        let old_value = grid.replace_cell((0, 1), 'x');
        assert_eq!(old_value, 'c');
        assert_grid_equal(&grid, &Grid::new(2, 3, "abxdef".chars().collect()));
    }

    #[test]
    fn indices_test() {
        let grid: Grid<u32> = Grid::new_default(3, 2);
        let indices: Vec<GridIndex> = grid.indices().collect();
        assert_eq!(
            indices,
            vec![
                GridIndex::new(0, 0),
                GridIndex::new(1, 0),
                GridIndex::new(2, 0),
                GridIndex::new(0, 1),
                GridIndex::new(1, 1),
                GridIndex::new(2, 1)
            ]
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serialize_test() {
        let mut grid = small_example_grid();

        let json = serde_json::to_string(&grid).unwrap();

        println!("{}", json);
    }

    fn assert_grid_equal<T>(actual: &Grid<T>, expected: &Grid<T>)
    where
        T: Display + PartialEq + Debug,
    {
        println!("actual:");
        println!("{}", actual.to_pretty_string());
        println!("expected:");
        println!("{}", expected.to_pretty_string());
        assert_eq!(actual, expected);
    }
}
