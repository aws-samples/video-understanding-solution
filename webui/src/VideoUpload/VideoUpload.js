import React, { Component, useState } from "react";
import { PutObjectCommand } from "@aws-sdk/client-s3";

import Form from "react-bootstrap/Form";
import InputGroup from 'react-bootstrap/InputGroup';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";

import './VideoUpload.css';

export class VideoUpload extends Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedVideoFiles: null,
      isUploading: false,
    };

    this.s3Client = props.s3Client;
    this.bucketName = props.bucketName;
    this.rawFolder = props.rawFolder;
  }

  onFileChange = (event) => {
    // console.log(event.target.files[0])
    this.setState({
      selectedVideoFiles: event.target.files,
    });
  };

  onFilesUpload = async () => {
    this.setState({
      isUploading: true
    });

    for (let i = 0; i < this.state.selectedVideoFiles.length; i++) {
      const command = new PutObjectCommand({
        Bucket: this.bucketName,
        Key: `${this.rawFolder}/${this.state.selectedVideoFiles[i].name}`,
        Body: this.state.selectedVideoFiles[i],
      });
  
      // Upload file  
      try { 
        await this.s3Client.send(command);
      } catch (error) {
        console.log("Unexpected error: " + error)
      }
    }

    this.setState({
      isUploading: false,
      selectedVideoFiles: null
    });
  };

  render() {
    return (
      <>
        <Row>
          <Col xs={6}>
            <Form.Group controlId="formFile" className="mb-3 no-margin-bottom">
              <InputGroup>
                <InputGroup.Text id="video-upload-text">Upload video files</InputGroup.Text>
                <Form.Control
                  type="file"
                  accept="video/mp4"
                  onChange={this.onFileChange}
                  multiple={true}
                />
                <Button
                  onClick={this.onFilesUpload.bind(this)}
                  variant={!this.state.isUploading ? 'success' : 'primary'}
                  id="upload-video"
                  disabled={!this.state.selectedVideoFiles || this.state.isUploading}
                >
                  {this.state.isUploading ? 'Uploading . . .' : 'Upload'}
                </Button>
              </InputGroup>
            </Form.Group>
          </Col>
          <Col xs={6}>
            <p className="no-margin-bottom" align="left">Alternatively, upload to Amazon S3 bucket <i>"{this.bucketName}"</i> inside folder <i>"{this.rawFolder}"</i>.</p>
          </Col>
        </Row>
      </>
    );
  }
}
