
package com.example.search;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SearchIntegrationTest {

    @Test
    void shouldSearchSuccessfully() {

        SearchService service = new SearchService();

        var results = service.search("AI search");

        // Validate the complete service behavior
        assertNotNull(results);
        assertEquals(2, results.size());
        assertEquals("Machine Learning", results.get(0));
    }
}
