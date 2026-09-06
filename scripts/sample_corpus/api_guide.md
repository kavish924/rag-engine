# API Guide

## Authentication
All requests require an API key passed in the `Authorization` header as
a Bearer token. Keys can be generated from the dashboard under Settings.
Requests without a valid key return a 401 error.

## Rate Limits
The API enforces a limit of 100 requests per minute per API key. If you
exceed this, you'll receive a 429 response with a `Retry-After` header
indicating how many seconds to wait.

## Error Codes
- `400` — malformed request body
- `401` — missing or invalid API key
- `404` — resource not found
- `429` — rate limit exceeded
- `500` — internal server error, retry with exponential backoff
