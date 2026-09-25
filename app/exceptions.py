class AppError(Exception):

    def __init__(
        self,
        code: str,
        message: str,
    ):
        self.code = code
        self.message = message

        super().__init__(
            message
        )


class AIProviderUnavailableError(
    AppError
):

    def __init__(self):

        super().__init__(
            code="AI_PROVIDER_UNAVAILABLE",
            message=(
                "The AI service is temporarily unavailable."
            ),
        )


class AIProviderTimeoutError(
    AppError
):

    def __init__(self):

        super().__init__(
            code="AI_PROVIDER_TIMEOUT",
            message=(
                "The AI service timed out."
            ),
        )


class AIProviderResponseError(
    AppError
):

    def __init__(self):

        super().__init__(
            code="AI_PROVIDER_RESPONSE_ERROR",
            message=(
                "The AI service returned an invalid response."
            ),
        )


class URLFetchError(
    AppError
):

    def __init__(self):

        super().__init__(
            code="URL_FETCH_FAILED",
            message=(
                "The requested URL could not be retrieved."
            ),
        )


class PDFExtractionError(
    AppError
):

    def __init__(self):

        super().__init__(
            code="PDF_EXTRACTION_FAILED",
            message=(
                "The PDF content could not be extracted reliably."
            ),
        )