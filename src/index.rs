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

    /// Get the `GridIndex` above, if it exists (no `GridIndex` exists above row 0).
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::GridIndex;
    /// let row_5 = GridIndex::new(17, 5);
    /// assert_eq!(row_5.up(), Some(GridIndex::new(17, 4)));
    /// let row_0 = GridIndex::new(38, 0);
    /// assert_eq!(row_0.up(), None);
    /// ```
    pub fn up(&self) -> Option<Self> {
        if self.row() > 0 {
            Some(Self::new(self.column(), self.row() - 1))
        } else {
            None
        }
    }

    /// Get the `GridIndex` to the right.
    ///
    /// ## Notes
    /// Unlike `up` and `left`, this method does not return an `Option<GridIndex>`, since there is
    /// always a higher column value. It's unlikely that you will ever have a `Grid` with
    /// `usize::MAX` rows, but if you did, this method would overflow.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::GridIndex;
    /// let column_17 = GridIndex::new(17, 11);
    /// assert_eq!(column_17.right(), GridIndex::new(18, 11));
    /// ```
    pub fn right(&self) -> Self {
        Self::new(self.column() + 1, self.row())
    }

    /// Get the `GridIndex` below.
    ///
    /// ## Notes
    /// Unlike `up` and `left`, this method does not return an `Option<GridIndex>`, since there is
    /// always a higher row value. It's unlikely that you will ever have a `Grid` with `usize::MAX`
    /// rows, but if you did, this method would overflow.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::GridIndex;
    /// let row_15 = GridIndex::new(3, 15);
    /// assert_eq!(row_15.down(), GridIndex::new(3, 16));
    /// ```
    pub fn down(&self) -> Self {
        Self::new(self.column(), self.row() + 1)
    }

    /// Get the `GridIndex` to the left, if it exists (no `GridIndex` exists to the left of column
    /// 0).
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::GridIndex;
    /// let column_9 = GridIndex::new(9, 10);
    /// assert_eq!(column_9.left(), Some(GridIndex::new(8, 10)));
    /// let column_0 = GridIndex::new(0, 10);
    /// assert_eq!(column_0.left(), None);
    /// ```
    pub fn left(&self) -> Option<Self> {
        if self.column() > 0 {
            Some(Self::new(self.column() - 1, self.row()))
        } else {
            None
        }
    }

    /// Convert this GridIndex into a linear index in a Grid of the given width.
    ///
    /// ## Panics
    /// * If `self.column() >= width`
    pub(crate) fn to_linear_idx_in(self, width: usize) -> usize {
        if self.column() >= width {
            panic!(
                "can't convert {:?} to a linear index in a Grid of width {}",
                self, width
            );
        }
        self.row() * width + self.column()
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
pub(crate) enum LinearIndexError {
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
mod tests {
    use super::*;

    #[test]
    fn to_linear_idx_in_test() {
        let index = GridIndex::new(2, 3);
        let linear = index.to_linear_idx_in(7);
        assert_eq!(linear, 23);
    }
}
