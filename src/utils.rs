use crate::{Grid, GridIndex};

#[inline(always)]
pub(crate) fn rows<T>(grid: &Grid<T>) -> impl DoubleEndedIterator<Item = usize> {
    0..grid.height
}

#[inline(always)]
pub(crate) fn columns<T>(grid: &Grid<T>) -> impl DoubleEndedIterator<Item = usize> {
    0..grid.width
}

pub(crate) fn cells<T>(grid: &'_ Grid<T>) -> impl DoubleEndedIterator<Item = (usize, usize, GridIndex)> {
    _cells(grid.width, grid.height)
}

fn _cells(width: usize, height: usize) -> impl DoubleEndedIterator<Item = (usize, usize, GridIndex)> {
    (0..height).flat_map(move |row| {
        (0..width).map(move |column| (column, row, GridIndex::new(column, row)))
    })
}

#[inline(always)]
pub(crate) fn panic_if_index_out_of_bounds<T>(grid: &Grid<T>, index: GridIndex) {
    panic_if_row_out_of_bounds(grid, index.row());
    panic_if_column_out_of_bounds(grid, index.column());
}

#[inline(always)]
pub(crate) fn panic_if_row_is_empty<T>(row: &[T]) {
    if row.is_empty() {
        panic!("row can't be empty");
    }
}

#[inline(always)]
pub(crate) fn panic_if_column_is_empty<T>(column: &[T]) {
    if column.is_empty() {
        panic!("column can't be empty");
    }
}

#[inline(always)]
pub(crate) fn panic_if_row_out_of_bounds<T>(grid: &Grid<T>, row: usize) {
    if row >= grid.height {
        panic!(
            "row index out of bounds: the height is {} but the row index is {}",
            grid.height, row
        );
    }
}

#[inline(always)]
pub(crate) fn panic_if_column_out_of_bounds<T>(grid: &Grid<T>, column: usize) {
    if column >= grid.width {
        panic!(
            "column index out of bounds: the width is {} but the column index is {}",
            grid.width, column
        );
    }
}

#[inline(always)]
pub(crate) fn panic_if_column_length_is_not_equal_to_height<T>(
    grid: &Grid<T>,
    column_length: usize,
) {
    if column_length != grid.height {
        panic!(
            "invalid length of column: was {}, should be {}",
            column_length, grid.height
        );
    }
}

#[inline(always)]
pub(crate) fn panic_if_row_length_is_not_equal_to_width<T>(grid: &Grid<T>, row_length: usize) {
    if row_length != grid.width {
        panic!(
            "invalid length of row: was {}, should be {}",
            row_length, grid.width
        );
    }
}

#[inline(always)]
pub(crate) fn panic_if_empty<T>(grid: &Grid<T>) {
    if grid.is_empty() {
        panic!("matrix is empty");
    }
}

#[inline(always)]
pub(crate) fn panic_if_not_square<T>(grid: &Grid<T>) {
    if !grid.is_square() {
        panic!(
            "matrix is not square: has {} columns, {} rows",
            grid.width, grid.height
        );
    }
}
