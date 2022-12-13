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

    /// Returns an iterator over the cardinal and ordinal neighbors of `self`.
    ///
    /// Returns the neighbors in clockwise order: `[up, up_right, right, down_right, down, down_left, left, up_left]`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::GridIndex;
    /// let idx = GridIndex::new(0, 1);
    /// let neighbors: Vec<_> = idx.neighbors().collect();
    /// assert_eq!(neighbors, vec![
    ///     (0, 0).into(), // up
    ///     (1, 0).into(), // up_right
    ///     (1, 1).into(), // right
    ///     (1, 2).into(), // down_right
    ///     (0, 2).into(), // down
    ///                    // nothing to the left since `idx` has column=0
    /// ]);
    /// ```
    pub fn neighbors(self) -> impl Iterator<Item = Self> {
        use std::iter::once;
        once::<fn(&Self) -> Option<Self>>(Self::up as _)
            .chain(once(Self::up_right as _))
            .chain(once(Self::right as _))
            .chain(once(Self::down_right as _))
            .chain(once(Self::down as _))
            .chain(once(Self::down_left as _))
            .chain(once(Self::left as _))
            .chain(once(Self::up_left as _))
            .map(move |f| f(&self))
            .filter_map(|i| i)
    }

    /// Returns an iterator over the cardinal neighbors of `self`.
    ///
    /// Returns the neighbors in clockwise order: `[up, right, down, left]`.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::GridIndex;
    /// let idx = GridIndex::new(0, 1);
    /// let neighbors: Vec<_> = idx.cardinal_neighbors().collect();
    /// assert_eq!(neighbors, vec![
    ///     (0, 0).into(), // up
    ///     (1, 1).into(), // right
    ///     (0, 2).into(), // down
    ///                    // nothing to the left since `idx` has column=0
    /// ]);
    /// ```
    pub fn cardinal_neighbors(self) -> impl Iterator<Item = Self> {
        use std::iter::once;
        once::<fn(&Self) -> Option<Self>>(Self::up as _)
            .chain(once(Self::right as _))
            .chain(once(Self::down as _))
            .chain(once(Self::left as _))
            .map(move |f| f(&self))
            .filter_map(|i| i)
    }

    /// Get the `GridIndex` above, if it exists.
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

    /// Get the `GridIndex` to the right, if it exists.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::GridIndex;
    /// let column_17 = GridIndex::new(17, 11);
    /// assert_eq!(column_17.right(), Some(GridIndex::new(18, 11)));
    /// ```
    pub fn right(&self) -> Option<Self> {
        if let Some(right) = self.column().checked_add(1) {
            Some(Self::new(right, self.row()))
        } else {
            None
        }
    }

    /// Get the `GridIndex` below, if it exists.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::GridIndex;
    /// let row_15 = GridIndex::new(3, 15);
    /// assert_eq!(row_15.down(), Some(GridIndex::new(3, 16)));
    /// ```
    pub fn down(&self) -> Option<Self> {
        if let Some(down) = self.row().checked_add(1) {
            Some(Self::new(self.column(), down))
        } else {
            None
        }
    }

    /// Get the `GridIndex` to the left, if it exists.
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

    /// Get the `GridIndex` above and to the left, if it exists.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let column_5_row_4 = GridIndex::new(5, 4);
    /// assert_eq!(column_5_row_4.up_left(), Some(GridIndex::new(4, 3)));
    /// let column_0_row_4 = GridIndex::new(0, 4);
    /// assert_eq!(column_0_row_4.up_left(), None);
    /// ```
    pub fn up_left(&self) -> Option<Self> {
        self.up().map(|up| up.left()).flatten()
    }

    /// Get the `GridIndex` above and to the right, if it exists.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let column_5_row_4 = GridIndex::new(5, 4);
    /// assert_eq!(column_5_row_4.up_right(), Some(GridIndex::new(6, 3)));
    /// let column_5_row_0 = GridIndex::new(5, 0);
    /// assert_eq!(column_5_row_0.up_right(), None);
    /// ```
    pub fn up_right(&self) -> Option<Self> {
        if self.row() > 0 {
            Some(Self::new(self.column() + 1, self.row() - 1))
        } else {
            None
        }
    }

    /// Get the `GridIndex` below and to the right, if it exists.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let column_5_row_4 = GridIndex::new(5, 4);
    /// assert_eq!(column_5_row_4.down_right(), Some(GridIndex::new(6, 5)));
    /// ```
    pub fn down_right(&self) -> Option<Self> {
        if let (Some(right), Some(down)) = (self.column().checked_add(1), self.row().checked_add(1))
        {
            Some(Self::new(right, down))
        } else {
            None
        }
    }

    /// Get the `GridIndex` below and to the left, if it exists.
    ///
    /// ## Example
    /// ```rust
    /// # use simple_grid::{Grid, GridIndex};
    /// let column_5_row_4 = GridIndex::new(5, 4);
    /// assert_eq!(column_5_row_4.down_left(), Some(GridIndex::new(4, 5)));
    /// let column_0_row_0 = GridIndex::new(0, 0);
    /// assert_eq!(column_0_row_0.down_left(), None);
    /// ```
    pub fn down_left(&self) -> Option<Self> {
        if self.column() > 0 {
            Some(Self::new(self.column() - 1, self.row() + 1))
        } else {
            None
        }
    }

    /// Convert this GridIndex into a linear index in a Grid of the given width.
    ///
    /// ## Panics
    /// * If `self.column() >= width`
    pub(crate) fn to_linear_idx_in(&self, width: usize) -> usize {
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
