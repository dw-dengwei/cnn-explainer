/* global tf */

// import {allOutputsjson} from '../../test/allOutputs'
// import {model} from '../../test/model'
// import {inputImageArrayjson} from '../../test/inputImageArray'
// Network input image size
const networkInputSize = 64;

// Enum of node types
const nodeType = {
  INPUT: 'input',
  CONV: 'conv',
  POOL: 'pool',
  RELU: 'relu',
  FC: 'fc',
  FLATTEN: 'flatten'
}

class Node {
  /**
   * Class structure for each neuron node.
   * 
   * @param {string} layerName Name of the node's layer.
   * @param {int} index Index of this node in its layer.
   * @param {string} type Node type {input, conv, pool, relu, fc}. 
   * @param {number} bias The bias assocated to this node.
   * @param {number[]} output Output of this node.
   */
  constructor(layerName, index, type, bias, output) {
    this.layerName = layerName;
    this.index = index;
    this.type = type;
    this.bias = bias;
    this.output = output;

    // Weights are stored in the links
    this.inputLinks = [];
    this.outputLinks = [];
  }
}

class Link {
  /**
   * Class structure for each link between two nodes.
   * 
   * @param {Node} source Source node.
   * @param {Node} dest Target node.
   * @param {number} weight Weight associated to this link. It can be a number,
   *  1D array, or 2D array.
   */
  constructor(source, dest, weight) {
    this.source = source;
    this.dest = dest;
    this.weight = weight;
  }
}
const get_model = (model_id) => {
  return new Promise((resolve, reject) => {
    var httpRequest = new XMLHttpRequest();
    // httpRequest.open('POST', 'http://10.112.35.137:5001/get_model', false);
    httpRequest.open('POST', 'http://10.112.35.137:5005/init/' + model_id, false);
    httpRequest.send(); // JSON.stringify({"model": model_id})
    const res = httpRequest.responseText;
    // let model = JSON.parse(res);
    // return model;
  });
}
const get_feature_map = (img_path) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = img_path;
    img.onload = function() {
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const context = canvas.getContext("2d");
        context.drawImage(img, 0, 0, img.width, img.height);
        const base64 = canvas.toDataURL("image/png");

        const formData = new FormData();
        formData.append("image", base64);

        fetch("http://10.112.35.137:5005/get_feature_map", {
            method: "POST",
            body: formData
        }).then(response => response.json())
          .then(data => resolve(data))
          .catch(error => reject(error));
      };
  });
}

// allOutputs, model, inputImageArray
export const constructCNNFront = async (img_path, is_init=false, model_id=-1) => {
    // let model = get_model();
    if (is_init) {
      get_model(model_id);
    }
    let feature_map = await get_feature_map(img_path);
    // let inputImageArray =  get_input_image_array();
    // let allOutputs = allOutputsjson['allOutputs']
    // let inputImageArray = inputImageArrayjson['inputImageArray']
    console.log(feature_map);
    let model = feature_map["model"];
    let inputImageArray = feature_map["inputImageArray"];
    let allOutputs = feature_map["allOutputs"]

    let last = allOutputs.length - 1;
    let max_value = allOutputs[last][0];
    let max_i = 0;
    for (let i = 0; i < 10; ++i) {
      if (allOutputs[last][i] > max_value) {
        max_i = i;
        max_value = allOutputs[last][i];
      }
    }
    // model.layers[5] = JSON.parse(JSON.stringify(model.layers[7]));
    // model.layers[5].name = "conv_3_1";
    // model.layers.splice(2, 0, newLayer(model.layers[7], 'conv_1_5'), newLayer(model.layers[8], 'relu_1_5'))
    // allOutputs.splice(2, 0, JSON.parse(JSON.stringify(allOutputs[7])), JSON.parse(JSON.stringify(allOutputs[8])))
    
    // model.layers.splice(4, 2);
    // allOutputs.splice(4, 2);
    // let temp = [];
    // for (let i = 1; i < allOutputs.length; ++i) {
    //   if (i === 1) {
    //     for (let j = 1; j <= 7; ++j)
    //       temp.push(allOutputs[i]);
    //   }else {
    //     temp.push(allOutputs[i]);
    //   }
    // }
    // allOutputs = temp;

    let cnn = [];
  
    // Add the first layer (input layer)
    let inputLayer = [];
    let inputShape = [64, 64, 3] // model.layers[0].batchInputShape.slice(1);
  
    // First layer's three nodes' outputs are the channels of inputImageArray
    for (let i = 0; i < inputShape[2]; i++) {
      let node = new Node('input', i, nodeType.INPUT, 0, inputImageArray[i]);
      inputLayer.push(node);
    }
                                                                                                                     
    cnn.push(inputLayer);
    let curLayerIndex = 1;
  
    for (let l = 0; l < model.layers.length; l++) {
      let layer = model.layers[l];
      // Get the current output
      let outputs = allOutputs[l];
  
      let curLayerNodes = [];
      let curLayerType;
  
      // Identify layer type based on the layer name
      if (layer.name.includes('conv')) {
        curLayerType = nodeType.CONV;
      } else if (layer.name.includes('pool')) {
        curLayerType = nodeType.POOL;
      } else if (layer.name.includes('relu')) {
        curLayerType = nodeType.RELU;
      } else if (layer.name.includes('output')) {
        layer.name = 'output';
        curLayerType = nodeType.FC;
      } else if (layer.name.includes('flatten')) {
        curLayerType = nodeType.FLATTEN;
      } else {
        console.log('Find unknown type');
      }
  
      // Construct this layer based on its layer type
      switch (curLayerType) {
        case nodeType.CONV: {
          let biases = layer.bias;
          // The new order is [output_depth, input_depth, height, width]
          let weights = layer.kernel;
  
          // Add nodes into this layer
          for (let i = 0; i < outputs.length; i++) {
            let node = new Node(layer.name, i, curLayerType, biases[i],
              outputs[i]);
  
            // Connect this node to all previous nodes (create links)
            // CONV layers have weights in links. Links are one-to-multiple.
            for (let j = 0; j < cnn[curLayerIndex - 1].length; j++) {
              let preNode = cnn[curLayerIndex - 1][j];
              let curLink = new Link(preNode, node, weights[i][j]);
              preNode.outputLinks.push(curLink);
              node.inputLinks.push(curLink);
            }
            curLayerNodes.push(node);
          }
          break;
        }
        case nodeType.FC: {
          let biases = layer.bias;
          // The new order is [output_depth, input_depth]
          let weights = layer.kernel;
  
          // Add nodes into this layer
          for (let i = 0; i < outputs.length; i++) {
            let node = new Node(layer.name, i, curLayerType, biases[i],
              outputs[i]);
  
            // Connect this node to all previous nodes (create links)
            // FC layers have weights in links. Links are one-to-multiple.
  
            // Since we are visualizing the logit values, we need to track
            // the raw value before softmax
            let curLogit = 0;
            for (let j = 0; j < cnn[curLayerIndex - 1].length; j++) {
              let preNode = cnn[curLayerIndex - 1][j];
              let curLink = new Link(preNode, node, weights[i][j]);
              preNode.outputLinks.push(curLink);
              node.inputLinks.push(curLink);
              curLogit += preNode.output * weights[i][j];
            }
            curLogit += biases[i];
            node.logit = curLogit;
            curLayerNodes.push(node);
          }
  
          // Sort flatten layer based on the node TF index
          cnn[curLayerIndex - 1].sort((a, b) => a.realIndex - b.realIndex);
          break;
        }
        case nodeType.RELU:
        case nodeType.POOL: {
          // RELU and POOL have no bias nor weight
          let bias = 0;
          let weight = null;
  
          // Add nodes into this layer
          for (let i = 0; i < outputs.length; i++) {
            let node = new Node(layer.name, i, curLayerType, bias, outputs[i]);
  
            // RELU and POOL layers have no weights. Links are one-to-one
            let preNode = cnn[curLayerIndex - 1][i];
            let link = new Link(preNode, node, weight);
            // console.log(l, i, preNode)
            preNode.outputLinks.push(link);
            node.inputLinks.push(link);
  
            curLayerNodes.push(node);
          }
          break;
        }
        case nodeType.FLATTEN: {
          // Flatten layer has no bias nor weights.
          let bias = 0;
  
          for (let i = 0; i < outputs.length; i++) {
            // Flatten layer has no weights. Links are multiple-to-one.
            // Use dummy weights to store the corresponding entry in the previsou
            // node as (row, column)
            // The flatten() in tf2.keras has order: channel -> row -> column
            let preNodeWidth = cnn[curLayerIndex - 1][0].output.length,
              preNodeNum = cnn[curLayerIndex - 1].length,
              preNodeIndex = i % preNodeNum,
              preNodeRow = Math.floor(Math.floor(i / preNodeNum) / preNodeWidth),
              preNodeCol = Math.floor(i / preNodeNum) % preNodeWidth,
              // Use channel, row, colume to compute the real index with order
              // row -> column -> channel
              curNodeRealIndex = preNodeIndex * (preNodeWidth * preNodeWidth) +
                preNodeRow * preNodeWidth + preNodeCol;
            
            let node = new Node(layer.name, i, curLayerType,
                bias, outputs[i]);
            
            // TF uses the (i) index for computation, but the real order should
            // be (curNodeRealIndex). We will sort the nodes using the real order
            // after we compute the logits in the output layer.
            node.realIndex = curNodeRealIndex;
  
            let link = new Link(cnn[curLayerIndex - 1][preNodeIndex],
                node, [preNodeRow, preNodeCol]);
  
            cnn[curLayerIndex - 1][preNodeIndex].outputLinks.push(link);
            node.inputLinks.push(link);
  
            curLayerNodes.push(node);
          }
  
          // Sort flatten layer based on the node TF index
          curLayerNodes.sort((a, b) => a.index - b.index);
          break;
        }
        default:
          console.error('Encounter unknown layer type');
          break;
      }
  
      // Add current layer to the NN
      cnn.push(curLayerNodes);
      curLayerIndex++;
    }
  
    return [cnn, max_i];
  }