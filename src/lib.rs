#[cfg(feature = "linalg")]
pub mod linalg;

pub(crate) mod utils;

use utils::*;

/// A two-dimensional array, indexed with x-and-y-coordinates (columns and rows).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Grid<T> {
    /// The width of the grid (number of columns).
    pub width: usize,
    /// The height of the grid (number of rows).
    pub height: usize,
    /// The data of the grid, stored in a linear array of `width * height` length.
    data: Vec<T>,
}

impl<T> Grid<T> {
    /// Construct a new Grid.
    ///
    /// # Panics
    /// * If `width * height != data.len()`
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// // construct a 2x3 (width x height) grid of chars
    /// let grid = Grid::new(2, 3, "abcdef".chars().collect());
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    /// // e f
    /// ```
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

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Checks if the Grid is square (number of columns and rows is equal).
    ///
    /// Note: an empty Grid is not square (even though columns and rows is 0).
    pub fn is_square(&self) -> bool {
        !self.is_empty() && self.width == self.height
    }

    pub fn has_same_dimensions(&self, other: &Grid<T>) -> bool {
        self.width == other.width && self.height == other.height
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

    fn is_empty(&self) -> bool {
        let ans = self.width == 0 || self.height == 0;
        if ans {
            debug_assert!(self.width == 0);
            debug_assert!(self.height == 0);
        }
        ans
    }

    /// Returns the area (number of columns * number of rows) of the grid.
    pub fn area(&self) -> usize {
        self.width * self.height
    }

    /// Attempts to get a reference to the element at `idx`.
    ///
    /// Returns `None` if `idx` is out of bounds.
    pub fn get<I>(&self, idx: I) -> Option<&T>
    where
        GridIndex: From<I>,
    {
        let index: usize = self.linear_idx(GridIndex::from(idx)).ok()?;

        Some(&self.data()[index])
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
    /// Goes from left->right, top->bottom.
    pub fn cell_iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.data.iter()
    }

    /// Return an iterator over the columns in the row with index `row`.
    ///
    /// # Panics
    /// * If `row >= self.height`
    ///
    /// # Example
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
    /// # Panics
    /// * If `column >= self.width`
    ///
    /// # Example
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
    /// # Panics
    /// * If `row_contents.is_empty()`
    /// * If `row_contents.len() != self.width`
    /// * If `row >= self.height`
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.insert_row(1, "xx".chars().collect());
    /// assert_eq!(grid, Grid::new(2, 3, "abxxcd".chars().collect()));
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // x x
    /// // c d
    /// ```
    pub fn insert_row(&mut self, row: usize, row_contents: Vec<T>) {
        panic_if_row_is_empty(&row_contents);

        if self.is_empty() && row == 0 {
            // special case, if the grid is empty, we can insert a row of any width
            self.set_width(row_contents.len());
            self.set_height(1);
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

        let start_idx = GridIndex::linear_idx_in(self.width, GridIndex::new(0, row));

        for (elem, idx) in row_contents.into_iter().zip(start_idx..) {
            self.data.insert(idx, elem);
        }

        self.set_height(self.height + 1);
    }

    /// Replace the contents in a row.
    ///
    /// # Panics
    /// * If `row >= self.height`
    /// * If `data.len() != self.width`
    pub fn replace_row(&mut self, row: usize, data: Vec<T>) {
        panic_if_row_out_of_bounds(self,row);
        panic_if_row_length_is_not_equal_to_width(self,data.len());

        for (column, elem) in data.into_iter().enumerate() {
            self[(column, row)] = elem;
        }
    }

    /// Replace the contents in a column.
    ///
    /// # Panics
    /// * If `column >= self.width`
    /// * If `data.len() != self.height`
    pub fn replace_column(&mut self, column: usize, data: Vec<T>) {
        panic_if_column_out_of_bounds(self,column);
        panic_if_column_length_is_not_equal_to_height(self,data.len());

        for (row, elem) in data.into_iter().enumerate() {
            self[(column, row)] = elem;
        }
    }

    /// Remove row at `row`, shifting all rows with higher indices "upward" (row `n` becomes row `n-1`).
    ///
    /// # Panics
    /// * If `row >= self.height`
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.remove_row(1);
    /// assert_eq!(grid, Grid::new(2, 1, "ab".chars().collect()));
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// ```
    pub fn remove_row(&mut self, row: usize) {
        panic_if_row_out_of_bounds(self,row);

        let start_idx = self.linear_idx(GridIndex::new(0, row)).unwrap();

        self.data.drain(start_idx..start_idx + self.width);
        self.set_height(self.height - 1);

        if self.height == 0 {
            //  no rows remain, so the grid is empty
            self.set_width(0);
        }
    }

    /// Insert a column at index `column`, shifting all other columns to the right (column `n` becomes column `n+1` and so on).
    ///
    /// # Panics
    /// * If `column_contents.is_empty()`
    /// * If `column_contents.len() != self.height`
    /// * If `column >= self.width`
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.insert_column(1, "xx".chars().collect());
    /// assert_eq!(grid, Grid::new(3, 2, "axbcxd".chars().collect()));
    /// println!("{}", grid);
    /// // prints:
    /// // a x b
    /// // c x d
    /// ```
    pub fn insert_column(&mut self, column: usize, column_contents: Vec<T>) {
        panic_if_column_is_empty(&column_contents);

        if self.is_empty() && column == 0 {
            // special case, if the grid is empty, we can insert a column of any height
            self.set_height(column_contents.len());
            self.set_width(1);
            self.data = column_contents;
            return;
        }

        panic_if_column_length_is_not_equal_to_height(self,column_contents.len());

        if column > self.width {
            // for example, if the width of the grid is 1,
            // we still want to support adding a column at the furthest right
            panic!(
                "column insertion index (is {}) should be <= width (is {})",
                column, self.width
            );
        }

        let indices: Vec<usize> = (0..column_contents.len())
            .map(|row| GridIndex::linear_idx_in(self.width + 1, GridIndex::new(column, row)))
            .collect();

        for (elem, idx) in column_contents.into_iter().zip(indices.into_iter()) {
            self.data.insert(idx, elem);
        }

        self.set_width(self.width + 1);
    }

    /// Remove column at `column`, shifting all columns with higher indices "left" (column `n` becomes column `n-1`).
    ///
    /// # Panics
    /// * If `column >= self.width`
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(2, 2, "abcd".chars().collect());
    /// grid.remove_column(1);
    /// assert_eq!(grid, Grid::new(1, 2, "ac".chars().collect()));
    /// println!("{}", grid);
    /// // prints:
    /// // a
    /// // c
    /// ```
    pub fn remove_column(&mut self, column: usize) {
        panic_if_column_out_of_bounds(self,column);

        let indices: Vec<usize> = (0..self.height)
            .map(|row| self.linear_idx(GridIndex::new(column, row)).unwrap())
            .collect();

        for idx in indices.into_iter().rev() {
            self.data.remove(idx);
        }

        self.set_width(self.width - 1);

        if self.width == 0 {
            //  no columns remain, so the grid is empty
            self.set_height(0);
        }
    }

    fn linear_idx(&self, idx: GridIndex) -> Result<usize, LinearIndexError> {
        if idx.row() >= self.height {
            Err(LinearIndexError::RowTooHigh)
        } else if idx.column() >= self.width {
            Err(LinearIndexError::ColumnTooHigh)
        } else {
            Ok(GridIndex::linear_idx_in(self.width, idx))
        }
    }
}

impl<T> Grid<T>
where
    T: Clone,
{
    /// Rotate the grid clockwise 90 degrees, cloning the data into a new grid.
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, "abcd".chars().collect());
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    ///
    /// let cw = grid.rotate_cw();
    /// assert_eq!(cw, Grid::new(2, 2, "cadb".chars().collect()));
    /// println!("{}", cw);
    /// // prints:
    /// // c a
    /// // d b
    /// ```
    pub fn rotate_cw(&self) -> Self {
        let mut rotated_data = Vec::with_capacity(self.area());

        for column in 0..self.width {
            for row in (0..self.height).rev() {
                rotated_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.height, self.width, rotated_data)
    }

    /// Rotate the grid counter-clockwise 90 degrees, cloning the data into a new grid.
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, "abcd".chars().collect());
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    ///
    /// let ccw = grid.rotate_ccw();
    /// assert_eq!(ccw, Grid::new(2, 2, "bdac".chars().collect()));
    /// println!("{}", ccw);
    /// // prints:
    /// // b d
    /// // a c
    /// ```
    pub fn rotate_ccw(&self) -> Self {
        let mut rotated_data = Vec::with_capacity(self.area());

        for column in (0..self.width).rev() {
            for row in 0..self.height {
                rotated_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.height, self.width, rotated_data)
    }

    /// Flip the grid horizontally, so that the first column becomes the last.
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, "abcd".chars().collect());
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    ///
    /// let hori = grid.flip_horizontally();
    /// assert_eq!(hori, Grid::new(2, 2, "badc".chars().collect()));
    /// println!("{}", hori);
    /// // prints:
    /// // b a
    /// // d c
    /// ```
    pub fn flip_horizontally(&self) -> Self {
        let mut flipped_data = Vec::with_capacity(self.area());

        for row in 0..self.height {
            for column in (0..self.width).rev() {
                flipped_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.width, self.height, flipped_data)
    }

    /// Flip the grid vertically, so that the first row becomes the last.
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, "abcd".chars().collect());
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    ///
    /// let vert = grid.flip_vertically();
    /// assert_eq!(vert, Grid::new(2, 2, "cdab".chars().collect()));
    /// println!("{}", vert);
    /// // prints:
    /// // c d
    /// // a b
    /// ```
    pub fn flip_vertically(&self) -> Self {
        let mut flipped_data = Vec::with_capacity(self.area());

        for row in (0..self.height).rev() {
            for column in 0..self.width {
                flipped_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.width, self.height, flipped_data)
    }

    /// Transpose the grid along the diagonal, so that cells at index (x, y) end up at index (y, x).
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 3, "abcdef".chars().collect());
    /// println!("{}", grid);
    /// // prints:
    /// // a b
    /// // c d
    /// // e f
    ///
    /// let transposed = grid.transpose();
    /// assert_eq!(transposed, Grid::new(3, 2, "acebdf".chars().collect()));
    /// println!("{}", transposed);
    /// // prints:
    /// // a c e
    /// // b d f
    /// ```
    pub fn transpose(&self) -> Self {
        let mut transposed_data = Vec::with_capacity(self.area());

        for column in 0..self.width {
            for row in 0..self.height {
                transposed_data.push(self[(column, row)].clone());
            }
        }

        Self::new(self.height, self.width, transposed_data)
    }
}

impl<T> Grid<T>
where
    T: Default,
{
    /// Create a grid filled with default values.
    pub fn new_default(width: usize, height: usize) -> Grid<T> {
        let data = (0..width * height).map(|_| T::default()).collect();
        Self::new(width, height, data)
    }
}

impl<T> Grid<T>
where
    T: PartialEq,
{
    /// Returns `true` if the grid contains some element equal to `value`.
    ///
    /// # Example
    /// ```
    /// # use simple_grid::Grid;
    /// let grid = Grid::new(2, 2, "abcd".chars().collect());
    /// assert!(grid.contains(&'a'));
    /// assert!(!grid.contains(&'e'));
    /// ```
    pub fn contains(&self, value: &T) -> bool {
        self.cell_iter().any(|element| element == value)
    }
}

impl<T> Grid<T>
where
    T: Copy,
{
    pub fn swap_rows(&mut self, row1: usize, row2: usize) {
        panic_if_row_out_of_bounds(self,row1);
        panic_if_row_out_of_bounds(self,row2);

        if row1 != row2 {
            for column in 0..self.width {
                let temp = self[(column, row1)];
                self[(column, row1)] = self[(column, row2)];
                self[(column, row2)] = temp;
            }
        }
    }

    pub fn swap_columns(&mut self, column1: usize, column2: usize) {
        panic_if_column_out_of_bounds(self,column1);
        panic_if_column_out_of_bounds(self,column2);

        if column1 != column2 {
            for row in 0..self.height {
                let temp = self[(column1, row)];
                self[(column1, row)] = self[(column2, row)];
                self[(column2, row)] = temp;
            }
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

impl<T, I> std::ops::Index<I> for Grid<T>
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

impl<T, I> std::ops::IndexMut<I> for Grid<T>
where
    GridIndex: From<I>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index: GridIndex = GridIndex::from(index);

        let linear = self.linear_idx(index).unwrap();

        &mut self.data[linear]
    }
}

impl<T> std::fmt::Display for Grid<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output = if let Some(max_length) = self.cell_iter().map(|c| c.to_string().len()).max() {
            let padded_string = |orig: &str| {
                let mut padding: String = std::iter::repeat(" ")
                    .take(max_length - orig.len())
                    .collect();
                padding.push_str(orig);
                padding
            };
            (0..self.height)
                .map(|r| {
                    (0..self.width)
                        .map(|c| padded_string(&self[(c, r)].to_string()))
                        .collect::<Vec<String>>()
                        .join(" ")
                })
                .collect::<Vec<String>>()
                .join("\n")
        } else {
            "".to_string()
        };
        write!(f, "{}", output)
    }
}

/// A struct used for indexing into a grid.
#[derive(Debug, Copy, Clone, PartialEq, Hash, Eq)]
pub struct GridIndex(usize, usize);

impl GridIndex {
    /// Construct a new GridIndex.
    pub fn new(column: usize, row: usize) -> Self {
        Self(column, row)
    }

    /// Get the column (x) index.
    pub fn column(&self) -> usize {
        self.0
    }

    /// Get the row (y) index.
    pub fn row(&self) -> usize {
        self.1
    }

    pub(crate) fn linear_idx_in(width: usize, idx: GridIndex) -> usize {
        idx.row() * width + idx.column()
    }
}

impl From<(usize, usize)> for GridIndex {
    fn from((column, row): (usize, usize)) -> Self {
        GridIndex::new(column, row)
    }
}

impl std::fmt::Display for GridIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.column(), self.row())
    }
}

#[derive(Debug, Copy, Clone)]
enum LinearIndexError {
    RowTooHigh,
    ColumnTooHigh,
}

impl std::fmt::Display for LinearIndexError {
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
        // 1   2   3   4   5   6   7   8   9  10
        // 11  12  13  14  15  16  17  18  19  20
        // 21  22  23  24  25  26  27  28  29  30
        // 31  32  33  34  35  36  37  38  39  40
        // 41  42  43  44  45  46  47  48  49  50
        // 51  52  53  54  55  56  57  58  59  60
        // 61  62  63  64  65  66  67  68  69  70
        // 71  72  73  74  75  76  77  78  79  80
        // 81  82  83  84  85  86  87  88  89  90
        // 91  92  93  94  95  96  97  98  99 100
        let grid = Grid::new(10, 10, (1..=100).collect());

        println!("Grid<u32>: ");
        println!("{}", grid);

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
        println!("{}", grid);

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
    }

    #[test]
    fn set_value_test() {
        let mut grid = small_example_grid();

        *grid.get_mut((0, 1)).unwrap() = 'x';

        assert_eq!(grid, Grid::new(2, 3, "abxdef".chars().collect()));

        grid[(0, 2)] = 'y';

        assert_eq!(grid, Grid::new(2, 3, "abxdyf".chars().collect()));
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

        assert_eq!(grid.data(), &vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]);
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

        assert_eq!(
            grid,
            Grid::new(2, 4, "abcdefxx".chars().collect::<Vec<_>>())
        );
    }

    #[test]
    fn remove_row_test() {
        let mut grid = small_example_grid();
        let items_in_row_1: Vec<char> = grid.row_iter(1).cloned().collect();

        assert_eq!(items_in_row_1, vec!['c', 'd']);
        assert_eq!(grid.height, 3);

        grid.remove_row(1);
        assert_eq!(grid.height, 2);
    }

    #[test]
    fn remove_row_until_empty_test() {
        let mut grid = small_example_grid();

        grid.remove_row(0);
        grid.remove_row(0);
        grid.remove_row(0);

        assert_eq!(grid, Grid::new(0, 0, Vec::new()));
        assert_eq!(grid.height, 0);
        assert_eq!(grid.width, 0);
        assert!(grid.is_empty());

        // since the grid is now empty, we can add a row of any non-zero length
        grid.insert_row(0, vec!['a', 'b', 'c']);

        assert_eq!(grid, Grid::new(3, 1, vec!['a', 'b', 'c']));
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

        assert_eq!(
            grid,
            Grid::new(3, 3, "abxcdxefx".chars().collect::<Vec<_>>())
        );
    }

    #[test]
    fn remove_column_test() {
        let mut grid = small_example_grid();
        let items_in_column_1: Vec<_> = grid.column_iter(1).cloned().collect();

        assert_eq!(items_in_column_1, vec!['b', 'd', 'f']);
        assert_eq!(grid.width, 2);

        grid.remove_column(1);
        assert_eq!(grid.width, 1);
    }

    #[test]
    fn remove_column_until_empty_test() {
        let mut grid = small_example_grid();

        grid.remove_column(0);
        grid.remove_column(0);

        assert_eq!(grid, Grid::new(0, 0, Vec::new()));
        assert_eq!(grid.height, 0);
        assert_eq!(grid.width, 0);
        assert!(grid.is_empty());

        // since the grid is now empty, we can add a column of any non-zero length
        grid.insert_column(0, vec!['a', 'b', 'c']);

        assert_eq!(grid, Grid::new(1, 3, vec!['a', 'b', 'c']));
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

    #[test]
    fn transpose_test() {
        let grid = small_example_grid();

        let transposed = grid.transpose();

        assert_eq!(transposed, Grid::new(3, 2, "acebdf".chars().collect()));

        let transposed_again = transposed.transpose();

        assert_eq!(grid, transposed_again);
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

        grid.replace_row(1, vec!['x', 'x']);

        assert_eq!(grid, Grid::new(2, 3, "abxxef".chars().collect()));
    }

    #[test]
    fn replace_column_test() {
        let mut grid = small_example_grid();

        grid.replace_column(0, vec!['x', 'x', 'x']);

        assert_eq!(grid, Grid::new(2, 3, "xbxdxf".chars().collect()));
    }

    #[test]
    fn swap_columns_test() {
        let mut grid = small_example_grid();

        grid.swap_columns(0, 1);

        assert_eq!(grid, Grid::new(2, 3, "badcfe".chars().collect()));
    }

    #[test]
    fn swap_rows_test() {
        let mut grid = small_example_grid();

        grid.swap_rows(1, 2);

        assert_eq!(grid, Grid::new(2, 3, "abefcd".chars().collect()));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serialize_test() {
        let mut grid = small_example_grid();

        let json = serde_json::to_string(&grid).unwrap();

        println!("{}", json);
    }
}
