use simple_grid::{Grid, GridIndex};
use std::{fmt::Display, option::Option};
use Color::*;

type Board = Grid<Cell>;

fn main() {
    let mut board: Board = Grid::new_default(7, 6);

    let mut current_player = Color::Yellow;
    loop {
        print_board(&board);
        println!("{} turn", current_player);
        let index = get_position_from_stdin(&board);
        if let Some(Cell::Empty) = board.get(index) {
            board[index] = Cell::Filled(current_player);
            current_player = current_player.opponent();
        }
        if let Some(winner) = is_over(&board) {
            match winner {
                Option::Some(player) => {
                    println!("{} wins!", player);
                    break;
                }
                Option::None => {
                    println!("it's a tie!");
                    break;
                }
            }
        }
    }
    print_board(&board);
}

fn print_board(board: &Board) {
    let header: String = format!(
        " {} ",
        (0..7).map(|c| format!("{c}")).collect::<Vec<_>>().join(" ")
    );
    println!("{header}");
    for row in board.rows() {
        let middle = board
            .row_iter(row)
            .map(|c| format!("{}", c))
            .collect::<Vec<_>>()
            .join("|");
        println!("|{}|", middle);
    }
    println!("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯");
}

fn get_position_from_stdin(board: &Board) -> GridIndex {
    let stdin = std::io::stdin();
    let read_line = || {
        let mut buffer = String::new();
        stdin.read_line(&mut buffer).unwrap();
        buffer.trim().to_owned()
    };
    loop {
        println!("input a column: ");
        if let Ok(column) = read_line().parse() {
            if let Some((row, _)) = board
                .column_iter(column)
                .rev()
                .enumerate()
                .find(|(_, c)| matches!(c, Cell::Empty))
            {
                return GridIndex::new(column, 5 - row);
            } else {
                println!("column {} is full", column);
            }
        } else {
            println!("invalid column");
        }
    }
}

fn is_over(board: &Board) -> Option<Option<Color>> {
    // check horizontals
    for row in board.rows() {
        let items_in_row: Vec<&_> = board.row_iter(row).collect();
        for four in items_in_row.windows(4) {
            if let Some(winner) = check_win(four) {
                return Some(Some(winner));
            }
        }
    }

    // check verticals
    for column in board.columns() {
        let items_in_column: Vec<&_> = board.column_iter(column).collect();
        for four in items_in_column.windows(4) {
            if let Some(winner) = check_win(four) {
                return Some(Some(winner));
            }
        }
    }

    // check diagonals
    for row in board.rows() {
        for column in board.columns() {
            if let Some(winner) = check_diagonal_win(board, GridIndex::new(column, row)) {
                return Some(Some(winner));
            }
        }
    }

    // is the board filled? nobody wins
    if board.cell_iter().all(|c| matches!(c, Cell::Filled(_))) {
        return Some(None);
    }

    None
}

fn check_win(window: &[&Cell]) -> Option<Color> {
    let red_win = window.iter().all(|c| matches!(c, Cell::Filled(Red)));
    if red_win {
        return Some(Red);
    }
    let yellow_win = window.iter().all(|c| matches!(c, Cell::Filled(Yellow)));
    if yellow_win {
        return Some(Yellow);
    }
    None
}

fn check_diagonal_win(board: &Board, start: GridIndex) -> Option<Color> {
    if start.column() >= 4 {
        return None;
    }

    // down-right
    if start.row() <= 2 {
        let cells: Vec<_> = (0..4)
            .map(|n| &board[(start.column() + n, start.row() + n)])
            .collect();

        if let Some(winner) = check_win(&cells) {
            return Some(winner);
        }
    }

    // up-right
    if start.row() >= 3 {
        let cells: Vec<_> = (0..4)
            .map(|n| &board[(start.column() + n, start.row() - n)])
            .collect();

        if let Some(winner) = check_win(&cells) {
            return Some(winner);
        }
    }

    None
}

enum Cell {
    Empty,
    Filled(Color),
}

impl Default for Cell {
    fn default() -> Self {
        Self::Empty
    }
}

impl Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output = match self {
            Cell::Empty => " ",
            Cell::Filled(Yellow) => "Y",
            Cell::Filled(Red) => "R",
        };

        write!(f, "{}", output)
    }
}

#[derive(Debug, Clone, Copy)]
enum Color {
    Red,
    Yellow,
}

impl Color {
    fn opponent(&self) -> Self {
        match self {
            Red => Yellow,
            Yellow => Red,
        }
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output = match self {
            Red => "Red",
            Yellow => "Yellow",
        };

        write!(f, "{}", output)
    }
}
