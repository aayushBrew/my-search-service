
package com.example.search;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SearchServiceTest {

    @Test
    void shouldReturnResults() {
        SearchService service = new SearchService();
        var results = service.search("machine learning");

        assertEquals(3, results.size());
    }

    @Test
    void shouldReturnEmptyForBlankQuery() {
        SearchService service = new SearchService();
        var results = service.search("");

        assertTrue(results.isEmpty());
    }
}
