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
