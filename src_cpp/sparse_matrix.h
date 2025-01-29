#pragma once
#include <vector>
#include <cassert>

class SparseMatrix {
public:
    // CSR format: values, column indices, and row pointers
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int rows;
    int cols;

    SparseMatrix(int r, int c) : rows(r), cols(c) {
        row_ptr.resize(rows + 1, 0);
    }

    // Matrix-vector multiplication
    std::vector<double> operator*(const std::vector<double>& vec) const {
        assert(vec.size() == cols);
        std::vector<double> result(rows, 0.0);
        
        for (int i = 0; i < rows; i++) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                result[i] += values[j] * vec[col_indices[j]];
            }
        }
        return result;
    }

    // Add value to position (i,j)
    void add(int i, int j, double value) {
        // Find if element exists
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
            if (col_indices[k] == j) {
                values[k] += value;
                return;
            }
        }
        
        // Element doesn't exist, need to insert it
        int insert_pos = row_ptr[i];
        while (insert_pos < row_ptr[i + 1] && col_indices[insert_pos] < j) {
            insert_pos++;
        }
        
        values.insert(values.begin() + insert_pos, value);
        col_indices.insert(col_indices.begin() + insert_pos, j);
        
        // Update row pointers
        for (int k = i + 1; k < row_ptr.size(); k++) {
            row_ptr[k]++;
        }
    }
}; 