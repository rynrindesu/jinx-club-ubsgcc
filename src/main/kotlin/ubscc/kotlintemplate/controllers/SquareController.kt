package ubscc.kotlintemplate.controllers

import ubscc.kotlintemplate.models.SquareCalcRequest
import org.slf4j.LoggerFactory
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController

@RestController
class SquareController {

    private val logger = LoggerFactory.getLogger(this.javaClass)

    @PostMapping("/square")
    fun calculateSquare(@RequestBody request: SquareCalcRequest): Int {
        logger.info("Request received $request")
        val result = request.input * request.input
        logger.info("Returning result $result")
        return result
    }
}
