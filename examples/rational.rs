use rational::Rational;
use simple_grid::{linalg::GaussianEliminationResult, Grid};

fn main() {
    let data: Vec<Rational> = [
        (2, 3),
        (-13, 2),
        (-1, 2),
        (4289, 396),
        (101, 100),
        (1, 1000),
        (-5, 1),
        (-5819743, 1287000),
        (90, 211),
        (100, 1),
        (255, 1719),
        (-52552578365_i64, 311204322),
    ]
    .iter()
    .map(|&t| Rational::from(t))
    .collect();

    let equation: Grid<Rational> = Grid::new(4, 3, data);

    println!("                     x                      y                      z");
    println!("--------------------------------------------------------------------");
    println!("{}", equation.to_pretty_string());

    match equation.clone().gaussian_elimination() {
        GaussianEliminationResult::InfiniteSolutions => println!("inf"),
        GaussianEliminationResult::SingleSolution(sol) => {
            let (x, y, z) = (sol[0], sol[1], sol[2]);
            println!("x = {x}");
            println!("y = {y}");
            println!("z = {z}");
            for row in equation.rows() {
                println!(
                    "({})*({}) + ({})*({}) + ({})*({}) = {}",
                    equation[(0, row)],
                    x,
                    equation[(1, row)],
                    y,
                    equation[(2, row)],
                    z,
                    equation[(3, row)]
                );
                assert_eq!(
                    x * equation[(0, row)] + y * equation[(1, row)] + z * equation[(2, row)],
                    equation[(3, row)]
                );
            }
        }
        GaussianEliminationResult::NoSolution => println!("none"),
    }
}
