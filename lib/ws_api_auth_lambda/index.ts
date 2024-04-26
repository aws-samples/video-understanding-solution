import { APIGatewayRequestAuthorizerHandler } from "aws-lambda";
import { CognitoJwtVerifier } from "aws-jwt-verify";

const UserPoolId = process.env.USER_POOL_ID!;
const AppClientId = process.env.APP_CLIENT_ID!;


export const handler: APIGatewayRequestAuthorizerHandler = async (
  event,
  _context,
) => {
  try {
// We will use the CognitoJwtVerifier to verify that the `idToken` 
// supplied by the client is indeed associated with our `UserPool`
    const verifier = CognitoJwtVerifier.create({
      userPoolId: UserPoolId,
      tokenUse: "id",
      clientId: AppClientId,
    });

    const encodedToken = event.queryStringParameters!.idToken!;
    const payload = await verifier.verify(encodedToken);

// Successfully Authenticated!
    console.log("Token is valid. Payload:", payload);
// Now return a policy which allows the client to connect!
    return allowPolicy(event.methodArn, payload);
  } catch (error: any) {
    console.log(error.message);
    return denyAllPolicy();
  }
};

function allowPolicy(methodArn: string, idToken: any) {
  return {
    principalId: idToken.sub,
    policyDocument: {
      Version: "2012-10-17",
      Statement: [
        {
          Action: "execute-api:Invoke",
          Effect: "Allow",
          Resource: methodArn,
        },
      ],
    },
    context: {
      // set userId in the context
      userId: idToken.sub,
    },
  };
}

function denyAllPolicy() {
  return {
    principalId: "*",
    policyDocument: {
      Version: "2012-10-17",
      Statement: [
        {
          Action: "*",
          Effect: "Deny",
          Resource: "*",
        },
      ],
    },
  };
}