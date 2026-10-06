## Authentication

- POST /auth/register                 : Register new user
- POST /auth/login                    : Login
- POST /auth/logout                   : Logout

## Conversations

- GET    /conversations/                 : Get all conversations
- GET    /conversations/{id}             : Get conversation by id
- POST   /conversations/                 : Create new conversation
- PATCH  /conversations/{id}             : Update conversation
- DELETE /conversations/{id}             : Delete conversation

## Messages

- GET    /conversations/{id}/messages        : Get all messages in a conversation
- GET    /conversations/{id}/messages/{id}   : Get message by id
- POST   /conversations/{id}/messages        : Create new message in a conversation
- PATCH  /conversations/{id}/messages/{id}   : Update message
- DELETE /conversations/{id}/messages/{id}   : Delete message

## Uploads

- GET    /conversations/{id}/uploads           : List all uploads
- GET    /conversations/{id}/uploads/{id}      : Get upload by id
- POST   /conversations/{id}/uploads           : Upload a file
- PATCH  /conversations/{id}/uploads/{id}      : Update upload
- DELETE /conversations/{id}/uploads/{id}      : Delete upload

## Artifacts

- GET    /conversations/{id}/artifacts         : List all artifacts
- GET    /conversations/{id}/artifacts/{id}    : Get artifact by id
- PATCH  /conversations/{id}/artifacts/{id}    : Update artifact
- DELETE /conversations/{id}/artifacts/{id}    : Delete artifact
