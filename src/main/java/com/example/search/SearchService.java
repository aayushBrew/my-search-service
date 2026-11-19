
package com.example.search;

import java.util.List;

public class SearchService {

    public List<String> search(String query) {

        if (query == null || query.isBlank()) {
            return List.of();
        }

        return List.of(
            "Machine Learning",
            "Search Engineering"
        );
    }
}