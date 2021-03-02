use crate::{utils::*, Grid};
use num_traits::{Float, Num};
use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Sub},
};

impl<T> Grid<T>
where
    T: Num + Copy,
{
    /// Generate the identity matrix of size `size`.
    ///
    /// # Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut g = Grid::identity(3);
    /// assert_eq!(g, Grid::new(3, 3, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]));
    /// println!("{}", g);
    /// // prints
    /// // 1 0 0
    /// // 0 1 0
    /// // 0 0 1
    /// ```
    ///
    /// # Panics
    /// * If `size == 0`
    pub fn identity(size: usize) -> Self {
        if size == 0 {
            panic!();
        }

        let mut data = Vec::new();

        for _ in 0..size - 1 {
            data.push(T::one());
            for _ in 0..size {
                data.push(T::zero());
            }
        }
        data.push(T::one());

        Self::new(size, size, data)
    }

    /// Removes a given column and row from the grid, returning the remaining grid.
    fn minor(&self, skip_column: usize, skip_row: usize) -> Grid<T> {
        let mut new_vec: Vec<T> = Vec::with_capacity((self.width - 1) * (self.height - 1));

        for row in self.rows() {
            for column in self.columns() {
                if row != skip_row && column != skip_column {
                    new_vec.push(self[(column, row)]);
                }
            }
        }

        Grid::new(self.width - 1, self.height - 1, new_vec)
    }

    /// Multiply all elements in a row by some factor.
    ///
    /// # Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut g = Grid::new(2, 2, vec![3, 8, 4, 6]);
    /// g.multiply_row(1, 5);
    /// assert_eq!(g[(0, 1)], 20);
    /// assert_eq!(g[(1, 1)], 30);
    /// ```
    ///
    /// # Panics
    /// * If `row` is out of bounds.
    pub fn multiply_row(&mut self, row: usize, factor: T) {
        panic_if_row_out_of_bounds(self, row);

        for column in self.columns() {
            self[(column, row)] = self[(column, row)] * factor;
        }
    }

    /// Perform the following row operation:
    /// 1. Take the contents of row `source_row`, multiplied by `factor`.
    /// 2. Add the result to row `target_row`.
    fn add_to_row(&mut self, target_row: usize, source_row: usize, factor: T) {
        panic_if_row_out_of_bounds(self, target_row);
        panic_if_row_out_of_bounds(self, source_row);

        for column in self.columns() {
            self[(column, target_row)] =
                self[(column, target_row)] + (self[(column, source_row)] * factor);
        }
    }

    /// Returns `true` if the matrix is symmetric (i.e. if its equal to its transpose).
    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() {
            false
        } else {
            for (column, row, _) in self.indices() {
                if self[(column, row)] != self[(row, column)] {
                    return false;
                }
            }
            true
        }
    }

    /// Returns the trace (sum of diagonals) of a square matrix.
    /// The trace of an empty matrix is 0.
    ///
    /// # Panics
    /// * If the grid is not square.
    pub fn trace(&self) -> T {
        panic_if_not_square(self);

        let mut sum = T::zero();
        for n in self.columns() {
            sum = sum + self[(n, n)];
        }

        sum
    }

    /// Calculate the determinant of a square `Grid`.
    ///
    /// # Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let two_by_two = Grid::new(2, 2, vec![3, 8, 4, 6]);
    /// assert_eq!(two_by_two.determinant(), -14);
    /// ```
    ///
    /// # Panics
    /// * If `self` is not a square grid.
    pub fn determinant(&self) -> T {
        if self.is_empty() {
            // determinant of an empty grid is 1
            return T::one();
        }
        panic_if_not_square(self);

        if self.dimensions() == (1, 1) {
            return self[(0, 0)];
        }

        let mut sum: Option<T> = None;

        for column in self.columns() {
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

    /// Returns `true` if all elements below the diagonal are zero.
    /// # Panics
    /// * If the grid is empty.
    /// * If the grid is not square.
    pub fn is_upper_triangular(&self) -> bool {
        panic_if_empty(self);
        panic_if_not_square(self);

        for row in self.rows() {
            for column in 0..row {
                if self[(column, row)] != T::zero() {
                    return false;
                }
            }
        }

        true
    }

    /// Returns `true` if all elements above the diagonal are zero.
    ///
    /// # Panics
    /// * If the grid is empty.
    /// * If the grid is not square.
    pub fn is_lower_triangular(&self) -> bool {
        panic_if_empty(self);
        panic_if_not_square(self);

        for column in self.columns() {
            for row in 0..column {
                if self[(column, row)] != T::zero() {
                    return false;
                }
            }
        }

        true
    }

    /// Returns `true` if all elements not on the diagonal are zero.
    ///
    /// # Panics
    /// * If the grid is empty.
    /// * If the grid is not square.
    pub fn is_triangular(&self) -> bool {
        self.is_upper_triangular() && self.is_lower_triangular()
    }

    /// Returns `true` if all elements in the grid are zero.
    pub fn is_zero(&self) -> bool {
        for (_, _, idx) in self.indices() {
            if self[idx] != T::zero() {
                return false;
            }
        }
        true
    }
}

impl<T> Grid<T>
where
    T: Copy + Float + Display + Debug,
{
    /// Check this grid and another for equality, using an epsilon.
    ///
    /// Useful when dealing with matrices containing floating point values.
    ///
    /// # Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let a = Grid::new(2, 2, vec![1.5, 2., -5., 0.333333333]);
    /// let b = Grid::new(2, 2, vec![3./2., 4.0_f64.sqrt(), -3.0 - 2.0, 1.0/3.0]);
    /// assert!(a.equal_by_epsilon(&b, 0.000000001));
    /// ```
    pub fn equal_by_epsilon(&self, other: &Grid<T>, epsilon: T) -> bool {
        if self.dimensions() != other.dimensions() {
            false
        } else {
            for (_, _, idx) in self.indices() {
                let diff = (self[idx] - other[idx]).abs();
                if diff > epsilon {
                    return false;
                }
            }
            true
        }
    }

    /// Finds the inverse (if it exists) for a square matrix.
    ///
    /// Requires the grid to be `mut`, because Gaussian elimination is performed alongside the identity matrix to generate the inverse.
    ///
    /// # Returns
    /// * `Some` if the inverse was found.
    /// * `None` if the grid has no inverse (the determinant is zero).
    ///
    /// # Example
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut invertible = Grid::new(3, 3, vec![3., 0., 2., 2., 0., -2., 0., 1., 1.]);
    /// let inverse = invertible.inverse().unwrap();
    /// assert!(inverse.equal_by_epsilon(&Grid::new(3, 3, vec![0.2, 0.2, 0., -0.2, 0.3, 1.0, 0.2, -0.3, 0.]), 1e-6));
    /// ```
    ///
    /// # Panics
    /// * If the grid is not square
    pub fn inverse(&mut self) -> Option<Grid<T>> {
        panic_if_not_square(self);
        if self.determinant() == T::zero() {
            return None;
        }

        let mut identity = Self::identity(self.width);
        for steps in self.columns() {
            // find leftmost non-zero column
            let col = match (steps..self.width)
                .find(|&c| !self.is_part_of_column_zero(c, steps, self.height - 1))
            {
                Some(col) => col,
                None => {
                    break;
                }
            };

            let row = (steps..self.height)
                .find(|&r| self[(col, r)] != T::zero())
                .unwrap();

            self.swap_rows(steps, row);
            identity.swap_rows(steps, row);

            // multiply row so that first element is 1
            let factor = T::one() / self[(col, steps)];
            self.multiply_row(steps, factor);
            identity.multiply_row(steps, factor);

            for r in steps + 1..self.height {
                let factor = -self[(col, r)];
                self.add_to_row(r, steps, factor);
                identity.add_to_row(r, steps, factor);
            }
        }

        for row in self.rows().rev() {
            let non_zero_col = match self.columns().find(|&c| self[(c, row)] != T::zero()) {
                Some(col) => col,
                None => {
                    continue;
                }
            };

            for r in 0..row {
                let factor = -self[(non_zero_col, r)];
                self.add_to_row(r, row, factor);
                identity.add_to_row(r, row, factor);
            }
        }

        Some(identity)
    }

    /// Perform a Gaussian elimination, treating the rightmost column as the solutions to linear equations represented by the other columns.
    ///
    /// # Example
    /// To solve the following system:
    ///
    /// `2x + y - z = 8`
    ///
    /// `-3x - y + 2z = -11`
    ///
    /// `-2x + y + 2z = -3`
    /// ```rust
    /// # use simple_grid::Grid;
    /// let mut grid = Grid::new(4, 3, vec![2., 1., -1., 8., -3., -1., 2., -11., -2., 1., 2., -3.]);
    /// let solution = grid.gaussian_elimination();
    /// assert_eq!(solution.unwrap_single_solution(), vec![2., 3., -1.])
    /// ```
    pub fn gaussian_elimination(&mut self) -> GaussianEliminationResult<T> {
        for steps in 0..self.width - 1 {
            // find leftmost non-zero column
            let col = match (steps..self.width - 1)
                .find(|&c| !self.is_part_of_column_zero(c, steps, self.height - 1))
            {
                Some(col) => col,
                None => {
                    break;
                }
            };

            let row = (steps..self.height)
                .find(|&r| self[(col, r)] != T::zero())
                .unwrap();

            self.swap_rows(steps, row);

            // multiply row so that first element is 1
            let factor = T::one() / self[(col, steps)];
            self.multiply_row(steps, factor);

            for r in steps + 1..self.height {
                let factor = -self[(col, r)];
                self.add_to_row(r, steps, factor);
            }
        }

        for row in self.rows().rev() {
            let non_zero_col = match (0..self.width - 1).find(|&c| self[(c, row)] != T::zero()) {
                Some(col) => col,
                None => {
                    continue;
                }
            };

            for r in 0..row {
                let factor = -self[(non_zero_col, r)];

                self.add_to_row(r, row, factor);
            }
        }

        for row in self.rows() {
            if (0..self.width - 1).all(|column| self[(column, row)] == T::zero()) {
                if self[(self.width - 1, row)] != T::zero() {
                    return GaussianEliminationResult::NoSolution;
                } else {
                    return GaussianEliminationResult::InfiniteSolutions;
                }
            }
        }

        let solutions = self.column_iter(self.width - 1).copied().collect();
        GaussianEliminationResult::SingleSolution(solutions)
    }

    /// Returns `true` if elements in the range `row_start..=row_end` in `column` are all zero.
    fn is_part_of_column_zero(&self, column: usize, row_start: usize, row_end: usize) -> bool {
        panic_if_column_out_of_bounds(self, column);
        panic_if_row_out_of_bounds(self, row_start);
        panic_if_row_out_of_bounds(self, row_end);

        for row in row_start..=row_end {
            if self[(column, row)] != T::zero() {
                return false;
            }
        }

        true
    }
}

impl<T> Mul for Grid<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Copy,
{
    type Output = Grid<T>;

    fn mul(self, rhs: Grid<T>) -> Self::Output {
        if self.width != rhs.height {
            panic!(
                "invalid matrix dimensions for multiplication, lhs: {} columns, rhs: {} rows",
                self.width, rhs.height
            );
        }

        let mut product_vec: Vec<T> = Vec::with_capacity(self.height * rhs.width);

        for lhs_row in self.rows() {
            for rhs_column in rhs.columns() {
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

        Grid::new(rhs.width, self.height, product_vec)
    }
}

impl<T> Mul<T> for Grid<T>
where
    T: num_traits::Num + Mul<T> + Copy,
{
    type Output = Grid<T>;

    fn mul(mut self, rhs: T) -> Self::Output {
        for (_, _, idx) in self.indices() {
            self[idx] = self[idx] * rhs;
        }
        self
    }
}

impl<'a, T> Add<Grid<T>> for Grid<T>
where
    T: Add<T, Output = T> + Copy,
{
    type Output = Grid<T>;

    fn add(self, rhs: Grid<T>) -> Self::Output {
        if self.dimensions() != rhs.dimensions() {
            panic!("invalid matrix dimensions for addition, lhs: {} columns, {} rows, rhs: {} columns, {} rows", 
                self.width,
                self.height,
                rhs.width,
                rhs.height);
        }

        let mut sum_vec = Vec::with_capacity(self.area());

        for (_, _, idx) in self.indices() {
            sum_vec.push(self[idx] + rhs[idx]);
        }
        Grid::new(self.width, self.height, sum_vec)
    }
}

impl<'a, T> Sub<Grid<T>> for Grid<T>
where
    T: Sub<T, Output = T> + Copy,
{
    type Output = Grid<T>;

    fn sub(self, rhs: Grid<T>) -> Self::Output {
        if self.dimensions() != rhs.dimensions() {
            panic!("invalid matrix dimensions for addition, lhs: {} columns, {} rows, rhs: {} columns, {} rows",
                self.width,
                self.height,
                rhs.width,
                rhs.height);
        }

        let mut sum_vec = Vec::with_capacity(self.area());

        for (_, _, idx) in self.indices() {
            sum_vec.push(self[idx] - rhs[idx]);
        }
        Grid::new(self.width, self.height, sum_vec)
    }
}

/// Represents the result of performing a Gaussian elimination on a matrix.
#[derive(Debug)]
pub enum GaussianEliminationResult<T> {
    /// The system has infinite solutions.
    InfiniteSolutions,
    /// The system has exactly one solution, contained in the variant.
    SingleSolution(Vec<T>),
    /// The system has no solutions.
    NoSolution,
}

impl<T> GaussianEliminationResult<T> {
    /// Unwrap the solution single solution to a linear equation, panicking if there were zero or infinite solutions.
    pub fn unwrap_single_solution(self) -> Vec<T> {
        match self {
            GaussianEliminationResult::InfiniteSolutions => {
                panic!("result has infinite solutions")
            }
            GaussianEliminationResult::SingleSolution(s) => s,
            GaussianEliminationResult::NoSolution => {
                panic!("result has no solutions")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `2x + y - z = 8`
    ///
    /// `-3x - y + 2z = -11`
    ///
    /// `-2x + y + 2z = -3`
    fn single_solution_grid() -> Grid<f64> {
        Grid::new(
            4,
            3,
            vec![2., 1., -1., 8., -3., -1., 2., -11., -2., 1., 2., -3.],
        )
    }

    /// `2x + 3y = 10`
    ///
    /// `2x + 3y = 12`
    fn no_solution_grid() -> Grid<f64> {
        Grid::new(3, 2, vec![2., 3., 10., 2., 3., 12.])
    }

    /// `1x - y + 2z = -3`
    ///
    /// `4x + 4y - 2z = 1`
    ///
    /// `-2x + 2y - 4z = 6`
    fn infinite_solutions_grid() -> Grid<f64> {
        Grid::new(
            4,
            3,
            vec![1., -1., 2., -3., 4., 4., -2., 1., -2., 2., -4., 6.],
        )
    }

    #[test]
    fn multiply_matrix_by_matrix_test() {
        let a = Grid::new(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let b = Grid::new(2, 3, vec![7, 8, 9, 10, 11, 12]);

        let product = a * b;

        assert_eq!(product, Grid::new(2, 2, vec![58, 64, 139, 154]));

        let a = Grid::new(1, 4, vec![0.0, 0.5, -2.1, 1.0001]);
        let b = Grid::new(4, 1, vec![0.005, 9.7, -10.1, 0.0]);

        let product = a * b;

        assert_eq!(
            product,
            Grid::new(
                4,
                4,
                vec![
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0025,
                    4.85,
                    -5.05,
                    0.0,
                    -0.0105,
                    -20.37,
                    21.21,
                    -0.0,
                    0.0050005,
                    9.70097,
                    -10.101009999999999,
                    0.0
                ]
            )
        );
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
        let empty: Grid<f32> = Grid::new(0, 0, vec![]);
        assert_eq!(empty.determinant(), 1.0);

        let two_by_two = Grid::new(2, 2, vec![3, 8, 4, 6]);
        assert_eq!(two_by_two.determinant(), -14);

        let three_by_three = Grid::new(3, 3, vec![6, 1, 1, 4, -2, 5, 2, 8, 7]);
        assert_eq!(three_by_three.determinant(), -306);

        let five_by_five = Grid::new(
            5,
            5,
            vec![
                0, 6, -2, -1, 5, 0, 0, 0, -9, -7, 0, 15, 35, 0, 0, 0, -1, -11, -2, 1, -2, -2, 3, 0,
                -2,
            ],
        );
        assert_eq!(five_by_five.determinant(), 2480);
    }

    #[test]
    fn determinant_float_test() {
        let g = Grid::new(
            3,
            3,
            vec![1.0, 2.5, -9.0, 3.7, 2.1, -1.11, 12.3, -81.17, -10.0],
        );

        assert_eq!(g.determinant(), 2882.6998);
    }

    #[test]
    fn is_triangular_test() {
        let g = Grid::new(3, 3, vec![1, 4, 1, 0, 6, 4, 0, 0, 1]);
        assert!(g.is_upper_triangular());

        let g = Grid::new(3, 3, vec![1, 0, 0, 2, 8, 0, 4, 9, 7]);
        assert!(g.is_lower_triangular());

        let g = Grid::new(3, 3, vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
        assert!(g.is_triangular());
    }

    #[test]
    fn identity_test() {
        let g = Grid::identity(1);
        assert_eq!(g, Grid::new(1, 1, vec![1]));

        let g = Grid::identity(4);
        assert_eq!(
            g,
            Grid::new(4, 4, vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        );
    }

    #[test]
    fn is_symmetric_test() {
        let g = Grid::new(3, 3, vec![1, 7, 3, 7, 4, 5, 3, 5, 6]);
        assert!(g.is_symmetric());
    }

    #[test]
    fn gaussian_elimination_test() {
        let mut single_solution = single_solution_grid();
        let result = single_solution.gaussian_elimination();
        let solution = result.unwrap_single_solution();
        assert_eq!(solution, vec![2., 3., -1.]);

        let mut no_solution = no_solution_grid();
        let result = no_solution.gaussian_elimination();
        assert!(
            matches!(result, GaussianEliminationResult::NoSolution),
            "actual: {:?}",
            result
        );

        let mut infinite_solutions = infinite_solutions_grid();
        let result = infinite_solutions.gaussian_elimination();
        println!("{}", infinite_solutions);
        assert!(
            matches!(result, GaussianEliminationResult::InfiniteSolutions),
            "actual: {:?}",
            result
        );
    }

    #[test]
    fn inverse_test() {
        let original = float_grid(3, 3, vec![3, 0, 2, 2, 0, -2, 0, 1, 1]);
        let mut invertible = original.clone();
        let inverse = invertible.inverse().unwrap();
        compare_float_grids(&invertible, &Grid::identity(3), 0.0000001);
        compare_float_grids(
            &inverse,
            &Grid::new(3, 3, vec![0.2, 0.2, 0., -0.2, 0.3, 1.0, 0.2, -0.3, 0.]),
            0.0000001,
        );

        let product = original * inverse;
        compare_float_grids(&product, &Grid::identity(3), 0.0000000001);

        let original = float_grid(
            4,
            4,
            vec![5, -5, 5, 6, 2, 1, 1, 2, -1, -1, 0, 1, 5, 1, 2, 1],
        );
        let mut invertible = original.clone();
        let inverse = invertible.inverse().unwrap();
        compare_float_grids(&invertible, &Grid::identity(4), 0.00000000001);
        compare_float_grids(
            &inverse,
            &(float_grid(
                4,
                4,
                vec![
                    -4, -8, 26, 14, 2, 8, -19, -9, 10, 16, -63, -29, -2, 0, 15, 5,
                ],
            ) * (1.0 / 8.0)),
            0.00001,
        );
    }

    fn float_grid<T>(width: usize, height: usize, data: Vec<T>) -> Grid<f64>
    where
        T: Into<f64>,
    {
        Grid::new(width, height, data.into_iter().map(|e| e.into()).collect())
    }

    fn compare_float_grids(actual: &Grid<f64>, expected: &Grid<f64>, epsilon: f64) {
        assert_eq!(actual.width, expected.width);
        assert_eq!(actual.height, expected.height);
        println!("actual: ");
        println!("{}", actual);
        println!("expected: ");
        println!("{}", expected);
        for row in 0..actual.height {
            for column in 0..actual.width {
                let actual = actual[(column, row)];
                let expected = expected[(column, row)];
                let diff = (actual - expected).abs();
                assert!(
                    diff < epsilon,
                    "actual: {}, expected: {}, at index: ({}, {})",
                    actual,
                    expected,
                    column,
                    row
                );
            }
        }
    }
}
