# Simple Grid

I noticed I kept reimplementing the same 2d-grid structure in many of my personal projects, so I decided to make it into a library. This data structure does not attempt to be the fastest or best implementation of a 2d-grid, but it's simple to use and has zero dependencies.

# Example usage
## Creating a grid and accessing its cells:
```rust
use simple_grid::Grid;

let grid = Grid::new(10, 10, (1..=100).collect::<Vec<u32>>());
assert_eq!(grid.get((5, 2)).unwrap(), &26);

println!("{}", grid.to_pretty_string());
// prints:
//  1   2   3   4   5   6   7   8   9  10
// 11  12  13  14  15  16  17  18  19  20
// 21  22  23  24  25  26  27  28  29  30
// 31  32  33  34  35  36  37  38  39  40
// 41  42  43  44  45  46  47  48  49  50
// 51  52  53  54  55  56  57  58  59  60
// 61  62  63  64  65  66  67  68  69  70
// 71  72  73  74  75  76  77  78  79  80
// 81  82  83  84  85  86  87  88  89  90
// 91  92  93  94  95  96  97  98  99 100
```

## Iterating over cells:
```rust
let grid = Grid::new(10, 10, (1..=100).collect::<Vec<u32>>());

let elements_in_row_3: Vec<u32> = grid.row_iter(3).copied().collect();
assert_eq!(
    elements_in_row_3,
    vec![31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
);

let elements_in_column_7: Vec<u32> = grid.column_iter(7).copied().collect();
assert_eq!(
    elements_in_column_7,
    vec![8, 18, 28, 38, 48, 58, 68, 78, 88, 98]
);
```

## Modifying contents
```rust
let mut grid = Grid::new(10, 10, (1..=100).collect::<Vec<u32>>());

// get a mutable reference to a cell
*grid.get_mut((8, 2)).unwrap() = 1000;
assert_eq!(grid.get((8, 2)).unwrap(), &1000);

// can also access directly via the index operator
grid[(5,5)] = 1001;
assert_eq!(grid.get((5, 5)).unwrap(), &1001);
```

## Serializing/deserializing
This is only available if the `serde` feature is enabled.

## Linear algebra
The `linalg` feature includes some methods that are useful for linear algebra:

### Matrix operations
```rust
let grid1 = Grid::new(2, 2, vec![1, 2, 3, 4]);
let grid2 = Grid::new(2, 2, vec![1, 0, 1, 0]);
let sum = grid1 + grid2;
assert_eq!(sum, Grid::new(2, 2, vec![2, 2, 4, 4]));
```