use std::ops::{Add, Mul, Sub};

use num_traits::PrimInt;

use crate::Grid;

impl<T> Grid<T>
where
    T: PrimInt,
{
    pub fn determinant(&self) -> T {
        if self.is_empty() {
            panic!("cannot calculate the determinant of an empty matrix");
        }
        if !self.is_square() {
            panic!("cannot calculate the determinant of a non-square matrix");
        }

        if (self.width(), self.height()) == (1, 1) {
            return self[(0, 0)];
        }

        let mut sum: Option<T> = None;

        for column in 0..self.width() {
            let scalar = self[(column, 0)];
            let minor = self.minor(column, 0);
            let product = scalar * minor.determinant();
            match (sum, column % 2 == 0) {
                (Some(s), true) => sum = Some(s + product),
                (Some(s), false) => sum = Some(s - product),
                (None, _) => sum = Some(product),
            }
        }

        sum.unwrap()
    }

    fn minor(&self, skip_column: usize, skip_row: usize) -> Grid<T> {
        let mut new_vec: Vec<T> = Vec::with_capacity((self.width() - 1) * (self.height() - 1));

        for row in 0..self.height() {
            for column in 0..self.width() {
                if row != skip_row && column != skip_column {
                    new_vec.push(self[(column, row)]);
                }
            }
        }

        Grid::new(self.width() - 1, self.height() - 1, new_vec)
    }
}

impl<'a, T> Mul<Grid<T>> for Grid<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Copy,
{
    type Output = Grid<T>;

    fn mul(self, rhs: Grid<T>) -> Self::Output {
        if self.width() != rhs.height() {
            panic!("invalid matrix dimensions for multiplication, lhs: {} columns, {} rows, rhs: {} columns, {} rows", self.width(), 
            self.height(), rhs.width(), rhs.height());
        }

        let mut product_vec: Vec<T> = Vec::with_capacity(self.height() * rhs.width());

        for lhs_row in 0..self.height() {
            for rhs_column in 0..rhs.width() {
                let mut sum = None;
                for (&l, &r) in self.row_iter(lhs_row).zip(rhs.column_iter(rhs_column)) {
                    match sum {
                        Some(s) => sum = Some(s + (l * r)),
                        None => sum = Some(l * r),
                    }
                }
                product_vec.push(sum.unwrap());
            }
        }

        Grid::new(rhs.width(), self.height(), product_vec)
    }
}

impl<T> Mul<T> for Grid<T>
where
    T: num_traits::PrimInt,
{
    type Output = Grid<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut product_vec = Vec::with_capacity(self.area());

        for row in 0..self.height() {
            for column in 0..self.width() {
                let product: T = self[(column, row)] * rhs;
                product_vec.push(product);
            }
        }

        Grid::new(self.width(), self.height(), product_vec)
    }
}

impl<'a, T> Add<Grid<T>> for Grid<T>
where
    T: Add<T, Output = T> + Copy,
{
    type Output = Grid<T>;

    fn add(self, rhs: Grid<T>) -> Self::Output {
        if self.width() != rhs.width() || self.height() != rhs.height() {
            panic!("invalid matrix dimensions for addition, lhs: {} columns, {} rows, rhs: {} columns, {} rows", self.width(), 
            self.height(), rhs.width(), rhs.height()    );
        }

        let mut sum_vec = Vec::with_capacity(self.area());

        for row in 0..self.height() {
            for column in 0..self.width() {
                sum_vec.push(self[(column, row)] + rhs[(column, row)]);
            }
        }

        Grid::new(self.width(), self.height(), sum_vec)
    }
}

impl<'a, T> Sub<Grid<T>> for Grid<T>
where
    T: Sub<T, Output = T> + Copy,
{
    type Output = Grid<T>;

    fn sub(self, rhs: Grid<T>) -> Self::Output {
        if self.width() != rhs.width() || self.height() != rhs.height() {
            panic!("invalid matrix dimensions for addition, lhs: {} columns, {} rows, rhs: {} columns, {} rows", self.width(), 
            self.height(), rhs.width(), rhs.height()    );
        }

        let mut sum_vec = Vec::with_capacity(self.area());

        for row in 0..self.height() {
            for column in 0..self.width() {
                sum_vec.push(self[(column, row)] - rhs[(column, row)]);
            }
        }

        Grid::new(self.width(), self.height(), sum_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_matrix_by_matrix_test() {
        let a = Grid::new(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let b = Grid::new(2, 3, vec![7, 8, 9, 10, 11, 12]);

        let product = a * b;

        assert_eq!(product, Grid::new(2, 2, vec![58, 64, 139, 154]));
    }

    #[test]
    fn multiply_matrix_by_number_test() {
        let a = Grid::new(3, 2, vec![1, 2, 3, 4, 5, 6]);

        let product: Grid<i32> = a * 2;

        assert_eq!(product, Grid::new(3, 2, vec![2, 4, 6, 8, 10, 12]));
    }

    #[test]
    fn add_matrix_and_matrix_test() {
        let a = Grid::new(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let b = Grid::new(3, 2, vec![7, 8, 9, 10, 11, 12]);

        let sum = a + b;

        assert_eq!(sum, Grid::new(3, 2, vec![8, 10, 12, 14, 16, 18]));
    }

    #[test]
    fn sub_matrix_and_matrix_test() {
        let a = Grid::new(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let b = Grid::new(3, 2, vec![12, 11, 10, 10, 11, 5]);

        let sum = a - b;

        assert_eq!(sum, Grid::new(3, 2, vec![-11, -9, -7, -6, -6, 1]));
    }

    #[test]
    fn determinant_test() {
        let a = Grid::new(2, 2, vec![3, 8, 4, 6]);

        assert_eq!(a.determinant(), -14);

        let b = Grid::new(3, 3, vec![6, 1, 1, 4, -2, 5, 2, 8, 7]);

        assert_eq!(b.determinant(), -306);
    }
}
