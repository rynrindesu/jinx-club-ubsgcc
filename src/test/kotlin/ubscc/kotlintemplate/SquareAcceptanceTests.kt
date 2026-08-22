package ubscc.kotlintemplate

import org.assertj.core.api.Assertions
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.client.TestRestTemplate
import org.springframework.http.HttpEntity
import org.springframework.http.HttpStatus
import ubscc.kotlintemplate.models.SquareCalcRequest

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class SquareAcceptanceTests {
    @Autowired
    lateinit var testRestTemplate: TestRestTemplate

    @Test
    fun `should return square`() {
        val request = SquareCalcRequest(input = 2)
        val response = testRestTemplate.postForEntity("/square", HttpEntity(request), Int::class.java)
        Assertions.assertThat(response.statusCode).isEqualTo(HttpStatus.OK)
        val result = response.body
        Assertions.assertThat(result).isNotNull()
        Assertions.assertThat(result).isEqualTo(4)
    }
}