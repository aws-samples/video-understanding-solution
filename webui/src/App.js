import "./App.css";
import "@aws-amplify/ui-react/styles.css";
import { VideoTable } from "./VideoTable/VideoTable";
import { VideoUpload } from "./VideoUpload/VideoUpload";
import awsExports from "./aws-exports";
import { Amplify } from "aws-amplify";
import { withAuthenticator } from "@aws-amplify/ui-react";
import type { WithAuthenticatorProps } from "@aws-amplify/ui-react";
import { fetchAuthSession } from "aws-amplify/auth";
import { fromCognitoIdentityPool } from "@aws-sdk/credential-providers";
import {
  CognitoIdentityClient,
  GetIdCommand,
} from "@aws-sdk/client-cognito-identity";
import { S3Client } from "@aws-sdk/client-s3";
import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";

import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Navbar from "react-bootstrap/Navbar";

Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: awsExports.aws_user_pools_id,
      userPoolClientId: awsExports.aws_user_pools_web_client_id,
    },
  },
});

const REGION = awsExports.aws_cognito_region;
const COGNITO_ID = `cognito-idp.${REGION}.amazonaws.com/${awsExports.aws_user_pools_id}`;

function App({ signOut, user }: WithAuthenticatorProps) {
  const getTs = async () => {
    return (await fetchAuthSession()).tokens;
  }

  const getCredentials = async (ts) => {
    //const t = (await fetchAuthSession()).tokens;
    let idT = (await ts).idToken.toString();

    const cognitoidentity = new CognitoIdentityClient({
      credentials: fromCognitoIdentityPool({
        clientConfig: { region: awsExports.aws_cognito_region },
        identityPoolId: awsExports.aws_cognito_identity_pool_id,
        logins: {
          [COGNITO_ID]: idT,
        },
      }),
    });
    return await cognitoidentity.config.credentials();
  };

  const ts = getTs();
  const crs = getCredentials(ts);
  const s3Client = new S3Client({
    region: REGION,
    credentials: crs,
  });

  const bedrockClient = new BedrockRuntimeClient({
    region: REGION,
    credentials: crs,
  });

  return (
    <div className="App" user={user} key="app-root">
      <Navbar expand="lg" className="bg-body-tertiary">
        <Container>
          <Navbar.Brand href="#">Video Understanding Solution</Navbar.Brand>
        </Container>
      </Navbar>
      <Container id="VideoTableContainer">
        <Row>
          <Col></Col>
          <Col xs={10}>
            <Row>
              <Col>
                <VideoUpload
                  s3Client={s3Client}
                  bucketName={awsExports.bucket_name}
                  rawFolder={awsExports.raw_folder}
                />
              </Col>
            </Row>
            <Row>
              <Col>
                <hr></hr>
              </Col>
            </Row>
            <Row>
              <Col>
                <VideoTable
                  bucketName={awsExports.bucket_name}
                  s3Client={s3Client}
                  bedrockClient={bedrockClient}
                  fastModelId={awsExports.fast_model_id}
                  balancedModelId={awsExports.balanced_model_id}
                  rawFolder={awsExports.raw_folder}
                  summaryFolder={awsExports.summary_folder}
                  videoScriptFolder={awsExports.video_script_folder}
                  videoCaptionFolder={awsExports.video_caption_folder}
                  entitySentimentFolder={awsExports.entity_sentiment_folder}
                  transcriptionFolder={awsExports.transcription_folder}
                  restApiUrl={awsExports.rest_api_url}
                  videosApiResource={awsExports.videos_api_resource}
                  cognitoTs={ts}
                ></VideoTable>
              </Col>
            </Row>
          </Col>
          <Col></Col>
        </Row>
      </Container>
    </div>
  );
}

export default withAuthenticator(App, { hideSignUp: true });
