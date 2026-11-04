# Instructions for ANLP A3

1. Ensure you go through the given `.gitkeep` files in each directory.
2. For Part 1, the command for testing the heldout set should be `bash scripts/part1/test_assn.sh --heldout`. The output file must be named and stored in `output/part1/heldout_result_{roll_number}.jsonl`. The output should be one `{"id", "text"}` per line.
3. Since you are not allowed to write any comments or docstrings in your code, You are allowed to make a separate doc, CHOICES.md for this (exactly named like this, in your root folder)
4. Your README.md should ONLY have commands to run your code and replicate results (no implementation details or description at all)
5. All your directories and subdirectories (except `src` and `docs`), should have separate `part1/` and `part2/` subdirectories
6. The heldout file path for Part 1 should be `data/part1/heldout`.

Submit only ONE report for the entire assignment, the report should go directly in `docs`


Will be updated as clarifications are
 ased.

For part A, can we first train and test the transformer, pure mamba and pure deltanet, and then push design.md to the repo for the hybrid design (to get insights into what each arch. does individually)? Or do we have to push it before starting anything at all in part A?

The design.md document is supposed to be made and pushed before starting with your hybrid implementation. You may implement, train, and test the pure versions of the models before that. However, ensure that your hybrid model implementation exists independent of the above model implementations for us to ensure that the requirements of the design doc are met.