import React, { Component, useState } from "react";
import { PutObjectCommand } from "@aws-sdk/client-s3";

import Form from "react-bootstrap/Form";
import Row from "react-bootstrap/Row";
import Button from "react-bootstrap/Button";

export class VideoUpload extends Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedVideoFile: null,
      isUploading: false,
    };

    this.s3Client = props.s3Client;
    this.bucketName = props.bucketName;
    this.rawFolder = props.rawFolder;
  }

  onFileChange = (event) => {
    // console.log(event.target.files[0])
    this.setState({
      selectedVideoFile: event.target.files[0],
    });
  };

  onFileUpload = async () => {
    this.setState({
      isUploading: true
    });

    const command = new PutObjectCommand({
      Bucket: this.bucketName,
      Key: `${this.rawFolder}/${this.state.selectedVideoFile.name}`,
      Body: this.state.selectedVideoFile,
    });

    // Upload file  
    try { 
      await this.s3Client.send(command);
    } catch (error) {
      console.log("Unexpected error: " + error)
    } finally {
      this.setState({
        isUploading: false
      });
    }
  };

  render() {
    return (
      <>
        <Row>
          <Form.Group controlId="formFile" className="mb-3">
            <Form.Label>Video Upload</Form.Label>
            <Form.Control
              type="file"
              accept="video/mp4"
              onChange={this.onFileChange}
            />
          </Form.Group>
        </Row>
        <Row>
          <Button
            onClick={this.onFileUpload.bind(this)}
            variant={!this.state.isUploading ? 'success' : 'primary'}
            id="upload-video"
            disabled={!this.state.selectedVideoFile || this.state.isUploading}
          >
            {this.state.isUploading ? 'Loading…' : 'Upload'}
          </Button>
        </Row>
      </>
    );
  }
}
