use crate::{Grid, GridIndex};

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

#[inline(always)]
pub(crate) fn panic_if_width_xor_height_is_zero(width: usize, height: usize) {
    if (width == 0) ^ (height == 0) {
        panic!("if either width or height is 0, both must be 0");
    }
}
