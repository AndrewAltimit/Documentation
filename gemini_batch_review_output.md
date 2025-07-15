Loaded cached credentials.
Attempt 1 failed with 5xx error. Retrying with backoff... GaxiosError: [{
  "error": {
    "code": 400,
    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
    "errors": [
      {
        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
        "domain": "global",
        "reason": "badRequest"
      }
    ],
    "status": "INVALID_ARGUMENT"
  }
}
]
    at Gaxios._request (/home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/node_modules/gaxios/build/src/gaxios.js:142:23)
    at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
    at async OAuth2Client.requestAsync (/home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/build/src/auth/oauth2client.js:429:18)
    at async CodeAssistServer.requestStreamingPost (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:81:21)
    at async CodeAssistServer.generateContentStream (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:23:23)
    at async retryWithBackoff (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/utils/retry.js:62:20)
    at async GeminiChat.sendMessageStream (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/core/geminiChat.js:303:36)
    at async runNonInteractive (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/dist/src/nonInteractiveCli.js:44:36)
    at async main (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/dist/src/gemini.js:164:5) {
  config: {
    url: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse',
    method: 'POST',
    params: { alt: 'sse' },
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'GeminiCLI/v22.16.0 (linux; x64) google-api-nodejs-client/9.15.1',
      Authorization: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      'x-goog-api-client': 'gl-node/22.16.0'
    },
    responseType: 'stream',
    body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
    signal: AbortSignal { aborted: false },
    paramsSerializer: [Function: paramsSerializer],
    validateStatus: [Function: validateStatus],
    errorRedactor: [Function: defaultErrorRedactor]
  },
  response: {
    config: {
      url: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse',
      method: 'POST',
      params: [Object],
      headers: [Object],
      responseType: 'stream',
      body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      signal: [AbortSignal],
      paramsSerializer: [Function: paramsSerializer],
      validateStatus: [Function: validateStatus],
      errorRedactor: [Function: defaultErrorRedactor]
    },
    data: '[{\n' +
      '  "error": {\n' +
      '    "code": 400,\n' +
      '    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",\n' +
      '    "errors": [\n' +
      '      {\n' +
      '        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",\n' +
      '        "domain": "global",\n' +
      '        "reason": "badRequest"\n' +
      '      }\n' +
      '    ],\n' +
      '    "status": "INVALID_ARGUMENT"\n' +
      '  }\n' +
      '}\n' +
      ']',
    headers: {
      'alt-svc': 'h3=":443"; ma=2592000,h3-29=":443"; ma=2592000',
      'content-length': '387',
      'content-type': 'application/json; charset=UTF-8',
      date: 'Mon, 14 Jul 2025 19:12:12 GMT',
      server: 'ESF',
      'server-timing': 'gfet4t7; dur=4105',
      vary: 'Origin, X-Origin, Referer',
      'x-content-type-options': 'nosniff',
      'x-frame-options': 'SAMEORIGIN',
      'x-xss-protection': '0'
    },
    status: 400,
    statusText: 'Bad Request',
    request: {
      responseURL: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse'
    }
  },
  error: undefined,
  status: 400,
  [Symbol(gaxios-gaxios-error)]: '6.7.1'
}
Attempt 2 failed with 5xx error. Retrying with backoff... GaxiosError: [{
  "error": {
    "code": 400,
    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
    "errors": [
      {
        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
        "domain": "global",
        "reason": "badRequest"
      }
    ],
    "status": "INVALID_ARGUMENT"
  }
}
]
    at Gaxios._request (/home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/node_modules/gaxios/build/src/gaxios.js:142:23)
    at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
    at async OAuth2Client.requestAsync (/home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/build/src/auth/oauth2client.js:429:18)
    at async CodeAssistServer.requestStreamingPost (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:81:21)
    at async CodeAssistServer.generateContentStream (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:23:23)
    at async retryWithBackoff (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/utils/retry.js:62:20)
    at async GeminiChat.sendMessageStream (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/core/geminiChat.js:303:36)
    at async runNonInteractive (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/dist/src/nonInteractiveCli.js:44:36)
    at async main (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/dist/src/gemini.js:164:5) {
  config: {
    url: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse',
    method: 'POST',
    params: { alt: 'sse' },
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'GeminiCLI/v22.16.0 (linux; x64) google-api-nodejs-client/9.15.1',
      Authorization: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      'x-goog-api-client': 'gl-node/22.16.0'
    },
    responseType: 'stream',
    body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
    signal: AbortSignal { aborted: false },
    paramsSerializer: [Function: paramsSerializer],
    validateStatus: [Function: validateStatus],
    errorRedactor: [Function: defaultErrorRedactor]
  },
  response: {
    config: {
      url: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse',
      method: 'POST',
      params: [Object],
      headers: [Object],
      responseType: 'stream',
      body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      signal: [AbortSignal],
      paramsSerializer: [Function: paramsSerializer],
      validateStatus: [Function: validateStatus],
      errorRedactor: [Function: defaultErrorRedactor]
    },
    data: '[{\n' +
      '  "error": {\n' +
      '    "code": 400,\n' +
      '    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",\n' +
      '    "errors": [\n' +
      '      {\n' +
      '        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",\n' +
      '        "domain": "global",\n' +
      '        "reason": "badRequest"\n' +
      '      }\n' +
      '    ],\n' +
      '    "status": "INVALID_ARGUMENT"\n' +
      '  }\n' +
      '}\n' +
      ']',
    headers: {
      'alt-svc': 'h3=":443"; ma=2592000,h3-29=":443"; ma=2592000',
      'content-length': '387',
      'content-type': 'application/json; charset=UTF-8',
      date: 'Mon, 14 Jul 2025 19:12:22 GMT',
      server: 'ESF',
      'server-timing': 'gfet4t7; dur=4106',
      vary: 'Origin, X-Origin, Referer',
      'x-content-type-options': 'nosniff',
      'x-frame-options': 'SAMEORIGIN',
      'x-xss-protection': '0'
    },
    status: 400,
    statusText: 'Bad Request',
    request: {
      responseURL: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse'
    }
  },
  error: undefined,
  status: 400,
  [Symbol(gaxios-gaxios-error)]: '6.7.1'
}
Attempt 3 failed with 5xx error. Retrying with backoff... GaxiosError: [{
  "error": {
    "code": 400,
    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
    "errors": [
      {
        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
        "domain": "global",
        "reason": "badRequest"
      }
    ],
    "status": "INVALID_ARGUMENT"
  }
}
]
    at Gaxios._request (/home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/node_modules/gaxios/build/src/gaxios.js:142:23)
    at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
    at async OAuth2Client.requestAsync (/home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/build/src/auth/oauth2client.js:429:18)
    at async CodeAssistServer.requestStreamingPost (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:81:21)
    at async CodeAssistServer.generateContentStream (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:23:23)
    at async retryWithBackoff (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/utils/retry.js:62:20)
    at async GeminiChat.sendMessageStream (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/core/geminiChat.js:303:36)
    at async runNonInteractive (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/dist/src/nonInteractiveCli.js:44:36)
    at async main (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/dist/src/gemini.js:164:5) {
  config: {
    url: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse',
    method: 'POST',
    params: { alt: 'sse' },
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'GeminiCLI/v22.16.0 (linux; x64) google-api-nodejs-client/9.15.1',
      Authorization: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      'x-goog-api-client': 'gl-node/22.16.0'
    },
    responseType: 'stream',
    body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
    signal: AbortSignal { aborted: false },
    paramsSerializer: [Function: paramsSerializer],
    validateStatus: [Function: validateStatus],
    errorRedactor: [Function: defaultErrorRedactor]
  },
  response: {
    config: {
      url: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse',
      method: 'POST',
      params: [Object],
      headers: [Object],
      responseType: 'stream',
      body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      signal: [AbortSignal],
      paramsSerializer: [Function: paramsSerializer],
      validateStatus: [Function: validateStatus],
      errorRedactor: [Function: defaultErrorRedactor]
    },
    data: '[{\n' +
      '  "error": {\n' +
      '    "code": 400,\n' +
      '    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",\n' +
      '    "errors": [\n' +
      '      {\n' +
      '        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",\n' +
      '        "domain": "global",\n' +
      '        "reason": "badRequest"\n' +
      '      }\n' +
      '    ],\n' +
      '    "status": "INVALID_ARGUMENT"\n' +
      '  }\n' +
      '}\n' +
      ']',
    headers: {
      'alt-svc': 'h3=":443"; ma=2592000,h3-29=":443"; ma=2592000',
      'content-length': '387',
      'content-type': 'application/json; charset=UTF-8',
      date: 'Mon, 14 Jul 2025 19:12:36 GMT',
      server: 'ESF',
      'server-timing': 'gfet4t7; dur=3496',
      vary: 'Origin, X-Origin, Referer',
      'x-content-type-options': 'nosniff',
      'x-frame-options': 'SAMEORIGIN',
      'x-xss-protection': '0'
    },
    status: 400,
    statusText: 'Bad Request',
    request: {
      responseURL: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse'
    }
  },
  error: undefined,
  status: 400,
  [Symbol(gaxios-gaxios-error)]: '6.7.1'
}
Attempt 4 failed with 5xx error. Retrying with backoff... GaxiosError: [{
  "error": {
    "code": 400,
    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
    "errors": [
      {
        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
        "domain": "global",
        "reason": "badRequest"
      }
    ],
    "status": "INVALID_ARGUMENT"
  }
}
]
    at Gaxios._request (/home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/node_modules/gaxios/build/src/gaxios.js:142:23)
    at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
    at async OAuth2Client.requestAsync (/home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/google-auth-library/build/src/auth/oauth2client.js:429:18)
    at async CodeAssistServer.requestStreamingPost (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:81:21)
    at async CodeAssistServer.generateContentStream (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/code_assist/server.js:23:23)
    at async retryWithBackoff (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/utils/retry.js:62:20)
    at async GeminiChat.sendMessageStream (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/node_modules/@google/gemini-cli-core/dist/src/core/geminiChat.js:303:36)
    at async runNonInteractive (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/dist/src/nonInteractiveCli.js:44:36)
    at async main (file:///home/miku/.nvm/versions/node/v22.16.0/lib/node_modules/@google/gemini-cli/dist/src/gemini.js:164:5) {
  config: {
    url: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse',
    method: 'POST',
    params: { alt: 'sse' },
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'GeminiCLI/v22.16.0 (linux; x64) google-api-nodejs-client/9.15.1',
      Authorization: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      'x-goog-api-client': 'gl-node/22.16.0'
    },
    responseType: 'stream',
    body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
    signal: AbortSignal { aborted: false },
    paramsSerializer: [Function: paramsSerializer],
    validateStatus: [Function: validateStatus],
    errorRedactor: [Function: defaultErrorRedactor]
  },
  response: {
    config: {
      url: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse',
      method: 'POST',
      params: [Object],
      headers: [Object],
      responseType: 'stream',
      body: '<<REDACTED> - See `errorRedactor` option in `gaxios` for configuration>.',
      signal: [AbortSignal],
      paramsSerializer: [Function: paramsSerializer],
      validateStatus: [Function: validateStatus],
      errorRedactor: [Function: defaultErrorRedactor]
    },
    data: '[{\n' +
      '  "error": {\n' +
      '    "code": 400,\n' +
      '    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",\n' +
      '    "errors": [\n' +
      '      {\n' +
      '        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",\n' +
      '        "domain": "global",\n' +
      '        "reason": "badRequest"\n' +
      '      }\n' +
      '    ],\n' +
      '    "status": "INVALID_ARGUMENT"\n' +
      '  }\n' +
      '}\n' +
      ']',
    headers: {
      'alt-svc': 'h3=":443"; ma=2592000,h3-29=":443"; ma=2592000',
      'content-length': '387',
      'content-type': 'application/json; charset=UTF-8',
      date: 'Mon, 14 Jul 2025 19:12:55 GMT',
      server: 'ESF',
      'server-timing': 'gfet4t7; dur=3949',
      vary: 'Origin, X-Origin, Referer',
      'x-content-type-options': 'nosniff',
      'x-frame-options': 'SAMEORIGIN',
      'x-xss-protection': '0'
    },
    status: 400,
    statusText: 'Bad Request',
    request: {
      responseURL: 'https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse'
    }
  },
  error: undefined,
  status: 400,
  [Symbol(gaxios-gaxios-error)]: '6.7.1'
}
[API Error: [{
  "error": {
    "code": 400,
    "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
    "errors": [
      {
        "message": "The input token count (1845185) exceeds the maximum number of tokens allowed (1048576).",
        "domain": "global",
        "reason": "badRequest"
      }
    ],
    "status": "INVALID_ARGUMENT"
  }
}
]]
