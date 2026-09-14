use crate::{
    WasmerClient,
    types::{
        AutobuildDeploymentSubscription, AutobuildDeploymentSubscriptionVariables,
        PackageVersionReadySubscription, PackageVersionReadySubscriptionVariables, Uuid,
    },
};
use anyhow::Context;
use async_tungstenite::tungstenite::client::IntoClientRequest;
use cynic::SubscriptionBuilder;
use graphql_ws_client::Subscription;
use reqwest::header::HeaderValue;
use std::future::IntoFuture;

pub async fn package_version_ready(
    client: &WasmerClient,
    package_version_id: &str,
) -> anyhow::Result<
    Subscription<
        cynic::StreamingOperation<
            PackageVersionReadySubscription,
            PackageVersionReadySubscriptionVariables,
        >,
    >,
> {
    let mut url = client.graphql_endpoint().clone();
    if url.scheme() == "http" {
        url.set_scheme("ws").unwrap();
    } else if url.scheme() == "https" {
        url.set_scheme("wss").unwrap();
    }

    let url = url.to_string();
    let mut req = url.into_client_request()?;

    req.headers_mut().insert(
        "Sec-WebSocket-Protocol",
        HeaderValue::from_str("graphql-transport-ws").unwrap(),
    );

    if let Some(token) = client.auth_token() {
        req.headers_mut().insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}"))?,
        );
    }

    req.headers_mut()
        .insert(reqwest::header::USER_AGENT, client.user_agent.clone());

    let (connection, _resp) = async_tungstenite::tokio::connect_async(req)
        .await
        .context("could not connect")?;

    let (client, actor) =
        graphql_ws_client::Client::build(SubscriptionConnection(connection)).await?;
    tokio::spawn(actor.into_future());

    let stream = client
        .subscribe(PackageVersionReadySubscription::build(
            PackageVersionReadySubscriptionVariables {
                package_version_id: cynic::Id::new(package_version_id),
            },
        ))
        .await?;

    Ok(stream)
}

pub async fn autobuild_deployment(
    client: &WasmerClient,
    build_id: &str,
) -> anyhow::Result<
    Subscription<
        cynic::StreamingOperation<
            AutobuildDeploymentSubscription,
            AutobuildDeploymentSubscriptionVariables,
        >,
    >,
> {
    let mut url = client.graphql_endpoint().clone();
    if url.scheme() == "http" {
        url.set_scheme("ws").unwrap();
    } else if url.scheme() == "https" {
        url.set_scheme("wss").unwrap();
    }

    let url = url.to_string();
    let mut req = url.into_client_request()?;

    req.headers_mut().insert(
        "Sec-WebSocket-Protocol",
        HeaderValue::from_str("graphql-transport-ws").unwrap(),
    );

    if let Some(token) = client.auth_token() {
        req.headers_mut().insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}"))?,
        );
    }

    req.headers_mut()
        .insert(reqwest::header::USER_AGENT, client.user_agent.clone());

    let (connection, _resp) = async_tungstenite::tokio::connect_async(req)
        .await
        .context("could not connect")?;

    let (client, actor) =
        graphql_ws_client::Client::build(SubscriptionConnection(connection)).await?;
    tokio::spawn(actor.into_future());

    let stream = client
        .subscribe(AutobuildDeploymentSubscription::build(
            AutobuildDeploymentSubscriptionVariables {
                build_id: Uuid(build_id.to_string()),
            },
        ))
        .await?;

    Ok(stream)
}

// graphql-ws-client's built-in adapter targets tungstenite 0.28. Keep the
// protocol boundary here so subscriptions can use the current transport.
struct SubscriptionConnection<S>(S);

impl<S> graphql_ws_client::Connection for SubscriptionConnection<S>
where
    S: futures::Stream<
            Item = Result<
                async_tungstenite::tungstenite::Message,
                async_tungstenite::tungstenite::Error,
            >,
        > + futures::Sink<async_tungstenite::tungstenite::Message>
        + Send
        + Unpin,
    <S as futures::Sink<async_tungstenite::tungstenite::Message>>::Error: std::fmt::Display,
{
    async fn receive(&mut self) -> Option<graphql_ws_client::Message> {
        use async_tungstenite::tungstenite::Message as Ws;
        use futures::StreamExt;
        use graphql_ws_client::Message;
        loop {
            match self.0.next().await? {
                Ok(Ws::Text(text)) => return Some(Message::Text(text.to_string())),
                Ok(Ws::Ping(_)) => return Some(Message::Ping),
                Ok(Ws::Pong(_)) => return Some(Message::Pong),
                Ok(Ws::Close(frame)) => {
                    return Some(Message::Close {
                        code: frame.as_ref().map(|f| f.code.into()),
                        reason: frame.map(|f| f.reason.to_string()),
                    });
                }
                Ok(Ws::Binary(_) | Ws::Frame(_)) => continue,
                Err(error) => {
                    tracing::warn!(%error, "subscription websocket receive failed");
                    return None;
                }
            }
        }
    }

    async fn send(
        &mut self,
        message: graphql_ws_client::Message,
    ) -> Result<(), graphql_ws_client::Error> {
        use async_tungstenite::tungstenite::{Message as Ws, protocol::CloseFrame};
        use futures::SinkExt;
        use graphql_ws_client::Message;
        let frame = match message {
            Message::Text(text) => Ws::Text(text.into()),
            Message::Ping => Ws::Ping(Default::default()),
            Message::Pong => Ws::Pong(Default::default()),
            Message::Close { code, reason } => {
                Ws::Close(code.zip(reason).map(|(code, reason)| CloseFrame {
                    code: code.into(),
                    reason: reason.into(),
                }))
            }
        };
        self.0
            .send(frame)
            .await
            .map_err(|error| graphql_ws_client::Error::Send(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::SubscriptionConnection;
    use async_tungstenite::{
        WebSocketStream,
        tokio::TokioAdapter,
        tungstenite::{Message as Ws, protocol::Role},
    };
    use futures::StreamExt;
    use graphql_ws_client::{Connection, Message};

    #[tokio::test]
    async fn subscription_transport_preserves_text_and_control_frames() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        let client =
            WebSocketStream::from_raw_socket(TokioAdapter::new(client_io), Role::Client, None)
                .await;
        let mut server =
            WebSocketStream::from_raw_socket(TokioAdapter::new(server_io), Role::Server, None)
                .await;
        let mut client = SubscriptionConnection(client);

        server.send(Ws::Text("result".into())).await.unwrap();
        assert!(matches!(client.receive().await, Some(Message::Text(text)) if text == "result"));
        client
            .send(Message::Text("subscribe".into()))
            .await
            .unwrap();
        assert_eq!(
            server.next().await.unwrap().unwrap(),
            Ws::Text("subscribe".into())
        );

        server.send(Ws::Ping(Default::default())).await.unwrap();
        assert!(matches!(client.receive().await, Some(Message::Ping)));
        client.send(Message::Pong).await.unwrap();
        assert_eq!(
            server.next().await.unwrap().unwrap(),
            Ws::Pong(Default::default())
        );

        client
            .send(Message::Close {
                code: Some(1000),
                reason: Some("done".into()),
            })
            .await
            .unwrap();
        let Ws::Close(Some(frame)) = server.next().await.unwrap().unwrap() else {
            panic!("expected close frame");
        };
        assert_eq!(u16::from(frame.code), 1000);
        assert_eq!(frame.reason.as_str(), "done");
    }
}
