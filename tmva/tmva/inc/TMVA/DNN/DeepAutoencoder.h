// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TDeepAutoEncoder                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Deep Neural Network                                                       *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Akshay Vashistha   <akshayvashistha1995@gmail.com> - JSSATE, Noida, India *
 *                                                                                *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_DNN_DEEPAUTOENCODER
#define TMVA_DNN_DEEPAUTOENCODER

#include "TMVA/DNN/DeepNet.h"
#include "TMVA/MsgLogger.h"



#include "TMVA/Tools.h"
#include "TXMLEngine.h"


#include <iostream>
#include <fstream>
#include <string>

using namespace TMVA::DNN::DAE;

class TDeepNet; 

namespace TMVA {
namespace DNN {
namespace DAE {


/** \class TDeepAutoEncoder

    Deep AutoEncoder class.

    This classs encapsulates the information for Deep AutoEncoder Networks.

    \tparam Architecture The Architecture type that holds the
    architecture-specific data types.
 */
template <typename Architecture_t, typename Layer_t = VGeneralLayer<Architecture_t>>
class TDeepAutoEncoder :public TMVA::DNN::TDeepNet<Architecture_t, Layer_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;


private:
   std::vector<Layer_t *> fLayers; ///< The layers consisting the DeepNet

   size_t fBatchSize;   ///< Batch size used for training and evaluation.
   size_t fInputDepth;  ///< The depth of the input.
   size_t fInputHeight; ///< The height of the input.
   size_t fInputWidth;  ///< The width of the input.

   size_t fBatchDepth;  ///< The depth of the batch used for training/testing.
   size_t fBatchHeight; ///< The height of the batch used for training/testing.
   size_t fBatchWidth;  ///< The width of the batch used for training/testing.

   std::vector<Matrix_t> fLocalWeights; ///< Storing weights of all layers
   std::vector<Matrix_t> fLocalHiddenBiases; ///< Storing hidden biases locally for all layers
   std::vector<Matrix_t> fLocalVisibleBiases; ///< Stroring visible biases locally for all layers

   bool fIsTraining; ///< Is the network training?

   ELossFunction fJ;      ///< The loss function of the network.
   EInitialization fI;    ///< The initialization method of the network.
   ERegularization fR;    ///< The regularization used for the network.
   Scalar_t fWeightDecay; ///< The weight decay factor.
   bool fWasPreTrained;   ///< If PreTrain was executed.
   Scalar_t fDropoutProbability; 
   Scalar_t fCorruptionLevel; 

   MsgLogger* fLogger;                     //! message logger

   MsgLogger& Log() const { return *fLogger; }

public:

   /*! Default Constructor */
   TDeepAutoEncoder();

   /*! Constructor */
   TDeepAutoEncoder(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t BatchDepth,
            size_t BatchHeight, size_t BatchWidth, ELossFunction fJ, EInitialization fI = EInitialization::kZero,
            ERegularization fR = ERegularization::kNone, Scalar_t fWeightDecay = 0.0, bool isTraining = false);

   /*! Copy-constructor */
   TDeepAutoEncoder(const TDeepAutoEncoder &);

   /*! Destructor */
   ~TDeepAutoEncoder();


   TCorruptionLayer<Architecture_t> *AddCorruptionLayer(size_t visibleUnits, size_t hiddenUnits,
                                                        Scalar_t dropoutProbability, Scalar_t corruptionLevel);

   /*! Function for adding Corruption Layer in the Deep Neural Network,
     *  when the layer is already created.  */
   void AddCorruptionLayer(TCorruptionLayer<Architecture_t> *corruptionLayer);

   /*! Function for adding Compression layer in the Deep Neural Network,
    *  with given number of visibleUnits and hiddenUnits. It compresses the input units
    *   taking weights and biases from prev layers. */
   TCompressionLayer<Architecture_t> *AddCompressionLayer(size_t visibleUnits, size_t hiddenUnits,
                                                          Scalar_t dropoutProbability, EActivationFunction f,
                                                          std::vector<Matrix_t> weights, std::vector<Matrix_t> biases);

   /*! Function for adding Compression Layer in the Deep Neural Network, when
    *  the layer is already created. */
   void AddCompressionLayer(TCompressionLayer<Architecture_t> *compressionLayer);

   /*! Function for adding Reconstruction layer in the Deep Neural Network,
    *  with given number of visibleUnits and hiddenUnits. It reconstructs the input units
    *  taking weights and biases from prev layers. Same corruptionLevel and dropoutProbability
    *  must be passed as in corruptionLayer. */
   TReconstructionLayer<Architecture_t> *AddReconstructionLayer(size_t visibleUnits, size_t hiddenUnits,
                                                                Scalar_t learningRate, EActivationFunction f,
                                                                std::vector<Matrix_t> weights,
                                                                std::vector<Matrix_t> biases, Scalar_t corruptionLevel,
                                                                Scalar_t dropoutProbability);

   /*! Function for adding Reconstruction Layer in the Deep Neural Network, when
    *  the layer is already created. */
   void AddReconstructionLayer(TReconstructionLayer<Architecture_t> *reconstructionLayer);

   /*! Function for adding logisticRegressionLayer in the Deep Neural Network,
    *  with given number of inputUnits and outputUnits. It classifies the outputUnits. */
   TLogisticRegressionLayer<Architecture_t> *AddLogisticRegressionLayer(size_t inputUnits, size_t outputUnits,
                                                                        size_t testDataBatchSize,
                                                                        Scalar_t learningRate);

   /*! Function for adding logisticRegressionLayer in the Deep Neural Network, when
    *  the layer is already created. */
   void AddLogisticRegressionLayer(TLogisticRegressionLayer<Architecture_t> *logisticRegressionLayer);

   /*! Function for initialization of the Neural Net. */
   void Initialize();

   /*! Function that executes the entire forward pass in the network. */
   //void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   /*! Function that executes the entire backward pass in the network. */
   //void Backward(std::vector<Matrix_t> input, const Matrix_t &groundTruth, const Matrix_t &weights);

   /* To train the Deep AutoEncoder network with required number of Corruption, Compression and Reconstruction
    * layers. */
   void PreTrain(std::vector<Matrix_t> &input, std::vector<size_t> numHiddenUnitsPerLayer, Scalar_t learningRate,
                 Scalar_t corruptionLevel, Scalar_t dropoutProbability, size_t epochs, EActivationFunction f,
                 bool applyDropout = false);

   /* To classify outputLabel in Deep AutoEncoder. Should be used after PreTrain if required.
    * Currently, it used Logistic Regression Layer. Otherwise we can use any other classification layer also.
   */
   void FineTune(std::vector<Matrix_t> &input, std::vector<Matrix_t> &testInput, std::vector<Matrix_t> &outputLabel,
                 size_t outputUnits, size_t testDataBatchSize, Scalar_t learningRate, size_t epochs);

   /* Functions used to predict Encoded outputs from trained weights.*/
   Matrix_t PredictEncodedOutput(Matrix_t& input);

   /* Functions used to predict Reconstructed output from trained weights.*/
   Matrix_t PredictDecodedOutput(Matrix_t& input);

   /* Used to write weights and biases to files*/
   void WriteToFile(size_t layer, size_t layerSize);

   /*! Prediction based on activations stored in the last layer. */
   //void Prediction(Matrix_t &predictions, EOutputFunction f) const;

   /*! Prediction for the given inputs, based on what network learned. */
   //void Prediction(Matrix_t &predictions, std::vector<Matrix_t> input, EOutputFunction f);

   /*! Print the Deep Net Info */
   void Print();

   void WriteToXML(const TString filepath); 

   void ReadFromXML(const TString filepath); 
   void WriteMatrixXML(void *parent, const char *name, const TMatrixT<Double_t> &X); 
   void ReadMatrixXML(void *xml, const char *name, TMatrixT<Double_t> &X);


   /*! Get the layer in the vector of layers at poistion i */
   inline Layer_t *GetLayerAt(size_t i) { return fLayers[i]; }
   inline const Layer_t *GetLayerAt(size_t i) const { return fLayers[i]; }

   /* Depth and the output width of the network. */
   inline size_t GetDepth() { return fLayers.size(); }
   inline size_t GetOutputWidth() { return fLayers.back()->GetWidth(); }

   /* Return a reference to the layers. */
   inline std::vector<Layer_t *> &GetLayers() { return fLayers; }
   inline const std::vector<Layer_t *> &GetLayers() const { return fLayers; }

   /*! Remove all layers from the network. */
   inline void Clear() { fLayers.clear(); }

   /*! Getters */
   inline size_t GetBatchSize() const { return fBatchSize; }
   inline size_t GetInputDepth() const { return fInputDepth; }
   inline size_t GetInputHeight() const { return fInputHeight; }
   inline size_t GetInputWidth() const { return fInputWidth; }

   inline size_t GetBatchDepth() const { return fBatchDepth; }
   inline size_t GetBatchHeight() const { return fBatchHeight; }
   inline size_t GetBatchWidth() const { return fBatchWidth; }

   inline double GetDropoutProbability() const { return fDropoutProbability; }
   inline double GetCorruptionLevel() const { return fCorruptionLevel; }

   const std::vector<Matrix_t> &GetLocalWeights() const { return fLocalWeights; }
   std::vector<Matrix_t> &GetLocalWeights() { return fLocalWeights; }

   const Matrix_t &GetLocalWeightsAt(size_t i) const { return fLocalWeights[i]; }
   Matrix_t &GetLocalWeightsAt(size_t i) { return fLocalWeights[i]; }

   const std::vector<Matrix_t> &GetLocalHiddenBiases() const { return fLocalHiddenBiases; }
   std::vector<Matrix_t> &GetLocalHiddenBiases() { return fLocalHiddenBiases; }

   const Matrix_t &GetLocalHiddenBiasesAt(size_t i) const { return fLocalHiddenBiases[i]; }
   Matrix_t &GetLocalHiddenBiasesAt(size_t i) { return fLocalHiddenBiases[i]; }

   const std::vector<Matrix_t> &GetLocalVisibleBiases() const { return fLocalVisibleBiases; }
   std::vector<Matrix_t> &GetLocalVisibleBiases() { return fLocalVisibleBiases; }

   const Matrix_t &GetLocalVisibleBiasesAt(size_t i) const { return fLocalVisibleBiases[i]; }
   Matrix_t &GetLocalVisibleBiasesAt(size_t i) { return fLocalVisibleBiases[i]; }


   inline bool IsTraining() const { return fIsTraining; }
   inline bool GetWasPreTrained() const {return fWasPreTrained;}

   inline ELossFunction GetLossFunction() const { return fJ; }
   inline EInitialization GetInitialization() const { return fI; }
   inline ERegularization GetRegularization() const { return fR; }
   inline Scalar_t GetWeightDecay() const { return fWeightDecay; }

   /*! Setters */
   inline void SetBatchSize(size_t batchSize) { fBatchSize = batchSize; }
   inline void SetInputDepth(size_t inputDepth) { fInputDepth = inputDepth; }
   inline void SetInputHeight(size_t inputHeight) { fInputHeight = inputHeight; }
   inline void SetInputWidth(size_t inputWidth) { fInputWidth = inputWidth; }
   inline void SetBatchDepth(size_t batchDepth) { fBatchDepth = batchDepth; }
   inline void SetBatchHeight(size_t batchHeight) { fBatchHeight = batchHeight; }
   inline void SetBatchWidth(size_t batchWidth) { fBatchWidth = batchWidth; }
   inline void SetLossFunction(ELossFunction J) { fJ = J; }
   inline void SetInitialization(EInitialization I) { fI = I; }
   inline void SetRegularization(ERegularization R) { fR = R; }
   inline void SetWeightDecay(Scalar_t weightDecay) { fWeightDecay = weightDecay; }
   inline void SetDropoutProbability(Scalar_t dropout) {fDropoutProbability = dropout; }
   inline void SetCorruptionLevel(Scalar_t corruptionLevel) {fCorruptionLevel = corruptionLevel; } 
};

//
//  Deep Autoencoder Class - Implementation
//
//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
TDeepAutoEncoder<Architecture_t, Layer_t>::TDeepAutoEncoder()
   : fLayers(), fBatchSize(0), fInputDepth(0), fInputHeight(0), fInputWidth(0), fBatchDepth(0), fBatchHeight(0),
     fBatchWidth(0), fJ(ELossFunction::kMeanSquaredError), fI(EInitialization::kZero), fR(ERegularization::kNone),
     fWeightDecay(0.0), fIsTraining(true), fWasPreTrained(false)
{
   fLogger = new MsgLogger("TDeepAutoEncoder", kINFO); 
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
TDeepAutoEncoder<Architecture_t, Layer_t>::TDeepAutoEncoder(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                            size_t batchDepth, size_t batchHeight, size_t batchWidth, ELossFunction J,
                                            EInitialization I, ERegularization R, Scalar_t weightDecay, bool isTraining)
   : fLayers(), fBatchSize(batchSize), fInputDepth(inputDepth), fInputHeight(inputHeight), fBatchDepth(batchDepth),
     fBatchHeight(batchHeight), fBatchWidth(batchWidth), fInputWidth(inputWidth), fJ(J), fI(I), fR(R),
     fWeightDecay(weightDecay), fIsTraining(isTraining), fWasPreTrained(false)
{
   fLogger = new MsgLogger("TDeepAutoEncoder", kINFO); 
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
TDeepAutoEncoder<Architecture_t, Layer_t>::TDeepAutoEncoder(const TDeepAutoEncoder &deepNet)
   : fLayers(), fBatchSize(deepNet.fBatchSize), fInputDepth(deepNet.fInputDepth), fInputHeight(deepNet.fInputHeight),
     fInputWidth(deepNet.fInputWidth), fBatchDepth(deepNet.fBatchDepth), fBatchHeight(deepNet.fBatchHeight),
     fBatchWidth(deepNet.fBatchWidth), fJ(deepNet.fJ), fI(deepNet.fI), fR(deepNet.fR),
     fWeightDecay(deepNet.fWeightDecay), fIsTraining(deepNet.fIsTraining), fWasPreTrained(false)
{
   fLogger = new MsgLogger("TDeepAutoEncoder", kINFO); 
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
TDeepAutoEncoder<Architecture_t, Layer_t>::~TDeepAutoEncoder()
{
   delete fLogger; 
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
TCorruptionLayer<Architecture_t> *TDeepAutoEncoder<Architecture_t, Layer_t>::AddCorruptionLayer(size_t visibleUnits,
                                                                                        size_t hiddenUnits,
                                                                                        Scalar_t dropoutProbability,
                                                                                        Scalar_t corruptionLevel)
{
   size_t batchSize = this->GetBatchSize();

   TCorruptionLayer<Architecture_t> *corruptionLayer =
      new TCorruptionLayer<Architecture_t>(batchSize, visibleUnits, hiddenUnits, dropoutProbability, corruptionLevel);
   fLayers.push_back(corruptionLayer);
   return corruptionLayer;
}
//______________________________________________________________________________

template <typename Architecture_t, typename Layer_t>
void TDeepAutoEncoder<Architecture_t, Layer_t>::AddCorruptionLayer(TCorruptionLayer<Architecture_t> *corruptionLayer)
{
   fLayers.push_back(corruptionLayer);
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
TCompressionLayer<Architecture_t> *TDeepAutoEncoder<Architecture_t, Layer_t>::AddCompressionLayer(
   size_t visibleUnits, size_t hiddenUnits, Scalar_t dropoutProbability, EActivationFunction f,
   std::vector<Matrix_t> weights, std::vector<Matrix_t> biases)
{
   size_t batchSize = this->GetBatchSize();

   TCompressionLayer<Architecture_t> *compressionLayer = new TCompressionLayer<Architecture_t>(
      batchSize, visibleUnits, hiddenUnits, dropoutProbability, f, weights, biases);
   fLayers.push_back(compressionLayer);
   return compressionLayer;
}
//______________________________________________________________________________

template <typename Architecture_t, typename Layer_t>
void TDeepAutoEncoder<Architecture_t, Layer_t>::AddCompressionLayer(TCompressionLayer<Architecture_t> *compressionLayer)
{
   fLayers.push_back(compressionLayer);
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
TReconstructionLayer<Architecture_t> *TDeepAutoEncoder<Architecture_t, Layer_t>::AddReconstructionLayer(
   size_t visibleUnits, size_t hiddenUnits, Scalar_t learningRate, EActivationFunction f, std::vector<Matrix_t> weights,
   std::vector<Matrix_t> biases, Scalar_t corruptionLevel, Scalar_t dropoutProbability)
{
   size_t batchSize = this->GetBatchSize();

   TReconstructionLayer<Architecture_t> *reconstructionLayer = new TReconstructionLayer<Architecture_t>(
      batchSize, visibleUnits, hiddenUnits, learningRate, f, weights, biases, corruptionLevel, dropoutProbability);
   fLayers.push_back(reconstructionLayer);
   return reconstructionLayer;
}
//______________________________________________________________________________

template <typename Architecture_t, typename Layer_t>
void TDeepAutoEncoder<Architecture_t, Layer_t>::AddReconstructionLayer(
   TReconstructionLayer<Architecture_t> *reconstructionLayer)
{
   fLayers.push_back(reconstructionLayer);
}

//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
TLogisticRegressionLayer<Architecture_t> *TDeepAutoEncoder<Architecture_t, Layer_t>::AddLogisticRegressionLayer(
   size_t inputUnits, size_t outputUnits, size_t testDataBatchSize, Scalar_t learningRate)
{
   size_t batchSize = this->GetBatchSize();

   TLogisticRegressionLayer<Architecture_t> *logisticRegressionLayer =
      new TLogisticRegressionLayer<Architecture_t>(batchSize, inputUnits, outputUnits, testDataBatchSize, learningRate);
   fLayers.push_back(logisticRegressionLayer);
   return logisticRegressionLayer;
}
//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
void TDeepAutoEncoder<Architecture_t, Layer_t>::AddLogisticRegressionLayer(
   TLogisticRegressionLayer<Architecture_t> *logisticRegressionLayer)
{
   fLayers.push_back(logisticRegressionLayer);
}


//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
auto TDeepAutoEncoder<Architecture_t, Layer_t>::Initialize() -> void
{
   for (size_t i = 0; i < fLayers.size(); i++) {
      fLayers[i]->Initialize();
   }
}

//_____________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
auto TDeepAutoEncoder<Architecture_t, Layer_t>::PreTrain(std::vector<Matrix_t> &input,
                                                         std::vector<size_t> numHiddenUnitsPerLayer, Scalar_t learningRate,
                                                         Scalar_t corruptionLevel, Scalar_t dropoutProbability, size_t epochs,
                                                         EActivationFunction f, bool applyDropout) -> void
{
   SetDropoutProbability(dropoutProbability); 
   SetCorruptionLevel(corruptionLevel); 
   std::vector<Matrix_t> inp1;
   std::vector<Matrix_t> inp2;
   inp1.emplace_back(1,1);
   inp2.emplace_back(1,1);
   size_t numOfHiddenLayers = numHiddenUnitsPerLayer.size();
   size_t batchSize = this->GetBatchSize();
   size_t visibleUnits = static_cast<size_t>(input[0].GetNrows());


   AddCorruptionLayer(visibleUnits, numHiddenUnitsPerLayer[0], dropoutProbability, corruptionLevel);
   fLayers.back()->Initialize();
   fLayers.back()->Forward(input, applyDropout);
   //fLayers.back()->Print();

   AddCompressionLayer(visibleUnits, numHiddenUnitsPerLayer[0], dropoutProbability, f, fLayers.back()->GetWeights(),
                       fLayers.back()->GetBiases());
   //fLayers.back()->Initialize();
   fLayers.back()->Forward(fLayers[fLayers.size() - 2]->GetOutput(), applyDropout); // as we have to pass corrupt input
   //fLayers.back()->Print();
   AddReconstructionLayer(visibleUnits, numHiddenUnitsPerLayer[0], learningRate, f, fLayers.back()->GetWeights(),
                          fLayers.back()->GetBiases(), corruptionLevel, dropoutProbability);
   //fLayers.back()->Initialize();
   fLayers.back()->Forward(fLayers[fLayers.size() - 2]->GetOutput(),
                           applyDropout); // as we have to pass compressed Input
   fLayers.back()->Backward(fLayers[fLayers.size() - 2]->GetOutput(), inp1, fLayers[fLayers.size() - 3]->GetOutput(),
                            input);
   //fLayers.back()->Print();
   // three layers are added, now pointer is on third layer
   size_t weightsSize = fLayers.back()->GetWeights().size();
   size_t biasesSize = fLayers.back()->GetBiases().size();
   for (size_t epoch = 0; epoch < epochs - 1; epoch++) {
      // fLayers[fLayers.size() - 3]->Forward(input,applyDropout);
      for (size_t j = 0; j < weightsSize; j++) {
         Architecture_t::Copy(fLayers[fLayers.size() - 2]->GetWeightsAt(j), fLayers.back()->GetWeightsAt(j));
      }
      for (size_t j = 0; j < biasesSize; j++) {
         Architecture_t::Copy(fLayers[fLayers.size() - 2]->GetBiasesAt(j), fLayers.back()->GetBiasesAt(j));
      }
      fLayers[fLayers.size() - 2]->Forward(fLayers[fLayers.size() - 3]->GetOutput(), applyDropout);
      //fLayers[fLayers.size() - 2]->Print();
      fLayers[fLayers.size() - 1]->Forward(fLayers[fLayers.size() - 2]->GetOutput(), applyDropout);
      fLayers[fLayers.size() - 1]->Backward(fLayers[fLayers.size() - 2]->GetOutput(), inp1,
                                            fLayers[fLayers.size() - 3]->GetOutput(), input);
   }
   //fLayers.back()->Print();

   for (size_t i = 1; i < numOfHiddenLayers; i++) {
      AddCorruptionLayer(numHiddenUnitsPerLayer[i - 1], numHiddenUnitsPerLayer[i], dropoutProbability, corruptionLevel);
      fLayers.back()->Initialize();
      fLayers.back()->Forward(fLayers[fLayers.size() - 3]->GetOutput(),
                              applyDropout); // as we have to pass compressed Input
      //fLayers.back()->Print();
      AddCompressionLayer(numHiddenUnitsPerLayer[i - 1], numHiddenUnitsPerLayer[i], dropoutProbability, f,
                          fLayers.back()->GetWeights(), fLayers.back()->GetBiases());
      //fLayers.back()->Initialize();
      fLayers.back()->Forward(fLayers[fLayers.size() - 2]->GetOutput(), applyDropout);
      //fLayers.back()->Print();
      AddReconstructionLayer(numHiddenUnitsPerLayer[i - 1], numHiddenUnitsPerLayer[i], learningRate, f,
                             fLayers.back()->GetWeights(), fLayers.back()->GetBiases(), corruptionLevel,
                             dropoutProbability);
      //fLayers.back()->Initialize();
      fLayers.back()->Forward(fLayers[fLayers.size() - 2]->GetOutput(),
                              applyDropout); // as we have to pass compressed Input
      fLayers.back()->Backward(fLayers[fLayers.size() - 2]->GetOutput(), inp1, fLayers[fLayers.size() - 3]->GetOutput(),
                               fLayers[fLayers.size() - 5]->GetOutput());
      //fLayers.back()->Print();
      // three layers are added, now pointer is on third layer
      size_t weightsSize = fLayers.back()->GetWeights().size();
      size_t biasesSize = fLayers.back()->GetBiases().size();
      for (size_t epoch = 0; epoch < epochs - 1; epoch++) {
         // fLayers[fLayers.size() - 3]->Forward(input,applyDropout);
         for (size_t j = 0; j < weightsSize; j++) {
            Architecture_t::Copy(fLayers[fLayers.size() - 2]->GetWeightsAt(j), fLayers.back()->GetWeightsAt(j));
         }
         for (size_t j = 0; j < biasesSize; j++) {
            Architecture_t::Copy(fLayers[fLayers.size() - 2]->GetBiasesAt(j), fLayers.back()->GetBiasesAt(j));
         }
         fLayers[fLayers.size() - 2]->Forward(fLayers[fLayers.size() - 3]->GetOutput(), applyDropout);
         fLayers[fLayers.size() - 1]->Forward(fLayers[fLayers.size() - 2]->GetOutput(), applyDropout);
         fLayers[fLayers.size() - 1]->Backward(fLayers[fLayers.size() - 2]->GetOutput(), inp1,
                                               fLayers[fLayers.size() - 3]->GetOutput(),
                                               fLayers[fLayers.size() - 5]->GetOutput());

      }

   }

   for(size_t layer=0; layer<numOfHiddenLayers; layer++)
   {
      this->GetLocalWeights().emplace_back(this->GetLayerAt( (3 * layer) + 2 )->GetWeightsAt(0).GetNrows(), this->GetLayerAt( (3 * layer) + 2 )->GetWeightsAt(0).GetNcols());
      this->GetLocalHiddenBiases().emplace_back(this->GetLayerAt( (3 * layer) + 2 )->GetBiasesAt(0).GetNrows(), 1);
      this->GetLocalVisibleBiases().emplace_back(this->GetLayerAt( (3 * layer) + 2 )->GetBiasesAt(1).GetNrows(), 1);

      Architecture_t::Copy(this->GetLocalWeightsAt(layer), this->GetLayerAt( (3 * layer) + 2 )->GetWeightsAt(0));
      Architecture_t::Copy(this->GetLocalHiddenBiasesAt(layer), this->GetLayerAt( (3 * layer) + 2 )->GetBiasesAt(0));
      Architecture_t::Copy(this->GetLocalVisibleBiasesAt(layer), this->GetLayerAt( (3 * layer) + 2 )->GetBiasesAt(1));

      this->WriteToFile(layer, numOfHiddenLayers);
   }

   fWasPreTrained = true;
}
//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
auto TDeepAutoEncoder<Architecture_t, Layer_t>::FineTune(std::vector<Matrix_t> &input, std::vector<Matrix_t> &testInput,
                                                 std::vector<Matrix_t> &inputLabel, size_t outputUnits,
                                                 size_t testDataBatchSize, Scalar_t learningRate, size_t epochs) -> void
{
   std::vector<Matrix_t> inp1;
   std::vector<Matrix_t> inp2;
   if (fLayers.size() == 0) // only classification Layer
   {
      size_t inputUnits = input[0].GetNrows();

      AddLogisticRegressionLayer(inputUnits, outputUnits, testDataBatchSize, learningRate);
      fLayers.back()->Initialize();
      for (size_t i = 0; i < epochs; i++) {
         fLayers.back()->Backward(inputLabel, inp1, input, inp2);
      }
      fLayers.back()->Forward(input, false);
      fLayers.back()->Print();
   } else { // if used after any other layer
      size_t inputUnits = fLayers.back()->GetOutputAt(0).GetNrows();
      AddLogisticRegressionLayer(inputUnits, outputUnits, testDataBatchSize, learningRate);
      fLayers.back()->Initialize();
      for (size_t i = 0; i < epochs; i++) {
         fLayers.back()->Backward(inputLabel, inp1, fLayers[fLayers.size() - 2]->GetOutput(), inp2);
      }
      fLayers.back()->Forward(testInput, false);
     // fLayers.back()->Print();
   }
}

template <typename Architecture_t, typename Layer_t>
typename Architecture_t::Matrix_t TDeepAutoEncoder<Architecture_t, Layer_t>::PredictEncodedOutput(Matrix_t& input)
{
   //std::cout << "Starting predict " << std::endl;
   if (fWasPreTrained == false)
   {
      Log() << kFATAL << "The autoencoder was not yet trained, unable to predict the output for the sample. " << Endl;
   }

   size_t size = this->GetLocalWeights().size();
   //std::cout<<"size is "<<size<<std::endl;
   Matrix_t output;
   for(size_t i=0; i<size; i++)
   {
      Matrix_t localOutput(this->GetLocalWeightsAt(i).GetNrows(), 1);
      Matrix_t weights(this->GetLocalWeightsAt(i).GetNrows(), this->GetLocalWeightsAt(i).GetNcols());
      Matrix_t biases(this->GetLocalHiddenBiasesAt(i).GetNrows(), 1);
      Architecture_t::EncodeInput(input, localOutput, this->GetLocalWeightsAt(i));
      Architecture_t::AddBiases(localOutput, this->GetLocalHiddenBiasesAt(i));
      //evaluate<Architecture_t>(localOutput, DNN::EActivationFunction::kSigmoid);
      input.ResizeTo(localOutput);
      Architecture_t::Copy(input,localOutput);
      if(i == size-1)
      {
        output.ResizeTo(localOutput);
        Architecture_t::Copy(output, localOutput);
      }
   }
   return output;
}
//______________________________________________________________________________

// input must be of transformed or compressed form.
template <typename Architecture_t, typename Layer_t>
typename Architecture_t::Matrix_t TDeepAutoEncoder<Architecture_t, Layer_t>::PredictDecodedOutput(Matrix_t& input)
{
   Matrix_t Input;
   Input.ResizeTo(input);
   Architecture_t::Copy(Input,input);
   //std::cout << "Starting predict " << std::endl;

   if (fWasPreTrained == false)
   {
      Log() << kFATAL << "The autoencoder was not yet trained, unable to predict the output for the sample. " << Endl;
   }

   size_t size = this->GetLocalWeights().size();

   Matrix_t output;
   for(size_t i= size ; i > 0; i--)
   {
      Matrix_t localOutput(this->GetLocalWeightsAt(i-1).GetNcols(), 1);
      Matrix_t weights(this->GetLocalWeightsAt(i-1).GetNrows(), this->GetLocalWeightsAt(i-1).GetNcols());
      Matrix_t biases(this->GetLocalVisibleBiasesAt(i-1).GetNrows(), 1);
      Architecture_t::ReconstructInput(Input, localOutput, this->GetLocalWeightsAt(i-1));
      Architecture_t::AddBiases(localOutput, this->GetLocalVisibleBiasesAt(i-1));

      Input.ResizeTo(localOutput);
      Architecture_t::Copy(Input,localOutput);

      if(i == 1)
      {
         output.ResizeTo(localOutput);
         Architecture_t::Copy(output, localOutput);
      }
   }
   return output;
}


//______________________________________________________________________________

template <typename Architecture_t, typename Layer_t>
auto TDeepAutoEncoder<Architecture_t, Layer_t>::WriteToFile(size_t layer, size_t totalLayers)
-> void
{
   std::ofstream weightsfile, hiddenbiasesfile, visiblebiasesfile, info;
   weightsfile.open("Weights"+std::to_string(layer)+".txt");
   hiddenbiasesfile.open("HiddenBiases"+std::to_string(layer)+".txt");
   visiblebiasesfile.open("VisibleBiases"+std::to_string(layer)+".txt");
   info.open("layersInfo.txt");
   info << totalLayers;
   for(size_t k=0; k<(size_t)this->GetLayerAt(3*layer+2)->GetWeightsAt(0).GetNrows(); k++)
   {
      hiddenbiasesfile << this->GetLayerAt(3*layer+2)->GetBiasesAt(0)(k,0)<<"\n";
      for(size_t m = 0; m<(size_t)this->GetLayerAt(3*layer+2)->GetWeightsAt(0).GetNcols(); m++)
      {
         weightsfile << this->GetLayerAt(3*layer+2)->GetWeightsAt(0)(k,m)<<"\n";
         visiblebiasesfile << this->GetLayerAt(3*layer+2)->GetBiasesAt(1)(m,0)<<"\n";
      }
      std::cout<<std::endl;
   }
   weightsfile.close();
   hiddenbiasesfile.close();
   visiblebiasesfile.close();
   info.close();
}


//______________________________________________________________________________
template <typename Architecture_t, typename Layer_t>
auto TDeepAutoEncoder<Architecture_t, Layer_t>::Print() -> void
{
   std::cout << "DEEP NEURAL NETWORK:" << std::endl;
   std::cout << "\t Loss function = " << static_cast<char>(this->GetLossFunction()) << std::endl;
   std::cout << "\t Network Depth = " << this->GetDepth() << std::endl;
   std::cout << "\t Input depth = " << this->GetInputDepth() << std::endl;
   std::cout << "\t Input height = " << this->GetInputHeight() << std::endl;
   std::cout << "\t Input width = " << this->GetInputWidth() << std::endl;
   std::cout << "\t Batch size = " << this->GetBatchSize() << std::endl;

   std::cout << "\t Layers: " << std::endl;

   for (size_t i = 0; i < fLayers.size(); i++) {
      fLayers[i]->Print();
   }
}

template <typename Architecture_t, typename Layer_t>
auto TDeepAutoEncoder<Architecture_t, Layer_t>::WriteToXML(const TString filepath) -> void {
  TString xmlfname(filepath);
  xmlfname.ReplaceAll( ".txt", ".xml" );
  if (fLogger) {Log() << kINFO  << "Creating xml weight file for DAE : " << gTools().Color("lightblue") << xmlfname << gTools().Color("reset") << Endl; }
  //std::cout << "Starting writing the file. " << std::endl; 
  void *doc      = gTools().xmlengine().NewDoc();
  void *rootnode = gTools().AddChild(0,"MethodSetup", "", true);
  gTools().xmlengine().DocSetRootElement(doc,rootnode);
  gTools().AddAttr(rootnode,"Method", "VariableDAETransform::TDeepAutoencoder");
  void *nn = gTools().xmlengine().NewChild(rootnode, 0, "Network"); 

  Int_t inputWidth = GetInputWidth();
  Int_t depth      = GetDepth();
  char  lossFunction = static_cast<char>(GetLossFunction());
  gTools().xmlengine().NewAttr(nn, 0, "Depth", gTools().StringFromInt(depth));
  gTools().xmlengine().NewAttr(nn, 0, "InputWidth", gTools().StringFromInt(inputWidth));
  gTools().xmlengine().NewAttr(nn, 0, "InputDepth", gTools().StringFromInt(GetInputDepth())); 
  gTools().xmlengine().NewAttr(nn, 0, "InputHeight", gTools().StringFromInt(GetInputHeight())); 
  gTools().xmlengine().NewAttr(nn, 0, "OutputWidth", gTools().StringFromInt(GetOutputWidth())); 
  gTools().xmlengine().NewAttr(nn, 0, "BatchSize", gTools().StringFromInt(GetBatchSize())); 
  gTools().xmlengine().NewAttr(nn, 0, "BatchWidth", gTools().StringFromInt(GetBatchWidth())); 
  gTools().xmlengine().NewAttr(nn, 0, "BatchHeight", gTools().StringFromInt(GetBatchHeight())); 
  gTools().xmlengine().NewAttr(nn, 0, "BatchDepth", gTools().StringFromInt(GetBatchDepth())); 
  gTools().xmlengine().NewAttr(nn, 0, "Initialization", TString(static_cast<char>(GetInitialization()))); 
  gTools().xmlengine().NewAttr(nn, 0, "LossFunction", TString(lossFunction));
  gTools().xmlengine().NewAttr(nn, 0, "Regularization", TString(static_cast<char>(GetRegularization()))); 
  gTools().xmlengine().NewAttr(nn, 0, "WeightDecay", TString(static_cast<char>(GetWeightDecay()))); 
  gTools().xmlengine().NewAttr(nn, 0, "DropoutProbability", gTools().StringFromDouble(GetDropoutProbability())); 
  gTools().xmlengine().NewAttr(nn, 0, "CorruptionLevel", gTools().StringFromDouble(GetCorruptionLevel())); 
  int wasPretrained = static_cast<int>(GetWasPreTrained()); 
  gTools().xmlengine().NewAttr(nn, 0, "WasPretrained",  gTools().StringFromInt(wasPretrained)); 

  for (Int_t i = 0; i < depth; i++) {
      const auto& layer = GetLayerAt(i);
      auto layerxml = gTools().xmlengine().NewChild(nn, 0, "Layer");
      gTools().xmlengine().NewAttr(layerxml, 0, "LayerType", gTools().StringFromInt(layer->GetType())); 
      int activationFunction = static_cast<int>(layer->GetActivationFunction());
      gTools().xmlengine().NewAttr(layerxml, 0, "ActivationFunction",
                                   TString::Itoa(activationFunction, 10));
      gTools().xmlengine().NewAttr(layerxml, 0, "VisibleUnits", gTools().StringFromInt(layer->GetInputUnits())); 
      gTools().xmlengine().NewAttr(layerxml, 0, "HiddenUnits", gTools().StringFromInt(layer->GetOutputUnits())); 
      //void *weights = gTools().xmlengine().NewChild(layerxml, 0, "Weights"); 
      for (unsigned int j=0; j<layer->GetWeights().size(); j++) {
        void *weights = gTools().xmlengine().NewChild(layerxml, 0, "InternalLayer"); 
        gTools().xmlengine().NewAttr(weights, 0, "Number", gTools().StringFromInt(j)); 
        WriteMatrixXML(weights, "Weights", layer->GetWeightsAt(j));  //std::to_string(j)
        WriteMatrixXML(weights, "Biases",  layer->GetBiasesAt(j));
      }
      
  }
  //std::cout << "Looped over layers. " << std::endl; 


   /*Log() << kINFO //<<Form("Dataset[%s] : ",DataInfo().GetName())
    << "Creating xml weight file: "
         << gTools().Color("lightblue") << xmlfname << gTools().Color("reset") << Endl;
   void* doc      = gTools().xmlengine().NewDoc();
   void* rootnode = gTools().AddChild(0,"MethodSetup", "", true);
   gTools().xmlengine().DocSetRootElement(doc,rootnode);
   gTools().AddAttr(rootnode,"Method", "VariableDAETransform"); */
      
  gTools().xmlengine().SaveDoc(doc,xmlfname);
  gTools().xmlengine().FreeDoc(doc);
}

template <typename Architecture_t, typename Layer_t>
auto TDeepAutoEncoder<Architecture_t, Layer_t>::WriteMatrixXML(void *parent,
                                      const char *name,
                                      const TMatrixT<Double_t> &X) -> void
{
   std::stringstream matrixStringStream("");
   matrixStringStream.precision( 16 );

   for (size_t i = 0; i < (size_t) X.GetNrows(); i++)
   {
      for (size_t j = 0; j < (size_t) X.GetNcols(); j++)
      {
         matrixStringStream << std::scientific << X(i,j) << " ";
      }
   }
   std::string s = matrixStringStream.str();
   void* matxml = gTools().xmlengine().NewChild(parent, 0, name);
   gTools().xmlengine().NewAttr(matxml, 0, "rows",
                                gTools().StringFromInt((int)X.GetNrows()));
   gTools().xmlengine().NewAttr(matxml, 0, "cols",
                                gTools().StringFromInt((int)X.GetNcols()));
   gTools().xmlengine().AddRawLine (matxml, s.c_str());
}

template <typename Architecture_t, typename Layer_t>
void TDeepAutoEncoder<Architecture_t, Layer_t>::ReadFromXML(const TString filepath) {
  TString tfname(filepath);

  Log() << kDEBUG << "Reading weight file: " << gTools().Color("lightblue") << tfname << gTools().Color("reset") << Endl;

  if (tfname.EndsWith(".xml") ) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,29,0)
      void* doc = gTools().xmlengine().ParseFile(tfname,gTools().xmlenginebuffersize()); // the default buffer size in TXMLEngine::ParseFile is 100k. Starting with ROOT 5.29 one can set the buffer size, see: http://savannah.cern.ch/bugs/?78864. This might be necessary for large XML files
#else
      void* doc = gTools().xmlengine().ParseFile(tfname);
#endif
      void* rootnode = gTools().xmlengine().DocGetRootElement(doc); // node "MethodSetup"

  auto netXML = gTools().GetChild(rootnode, "MethodSetup");
  if (!netXML){
     netXML = rootnode;    // Is this really what we want? 
  }

  auto nn = gTools().GetChild(netXML, "Network"); 

  size_t depth, inputWidth, inputDepth, inputHeight, outputWidth, batchSize, batchWidth, batchHeight, batchDepth;
  bool wasPretrained; 
  char initialization, lossFunction;
  int regularization, weightDecay; 
  double dropoutProbability, corruptionLevel; 
  gTools().ReadAttr(nn, "Depth", depth);
  gTools().ReadAttr(nn, "InputWidth", inputWidth);
  gTools().ReadAttr(nn, "InputDepth", inputDepth); 
  gTools().ReadAttr(nn, "InputHeight", inputHeight); 
  gTools().ReadAttr(nn, "OutputWidth", outputWidth); 
  gTools().ReadAttr(nn, "BatchSize", batchSize); 
  gTools().ReadAttr(nn, "BatchWidth", batchWidth); 
  gTools().ReadAttr(nn, "BatchHeight", batchHeight); 
  gTools().ReadAttr(nn, "BatchDepth", batchDepth); 
  gTools().ReadAttr(nn, "Initialization", initialization);
  gTools().ReadAttr(nn, "LossFunction", lossFunction);
  gTools().ReadAttr(nn, "Regularization", regularization); 
  gTools().ReadAttr(nn, "WeightDecay", weightDecay); 
  gTools().ReadAttr(nn, "DropoutProbability", dropoutProbability); 
  gTools().ReadAttr(nn, "CorruptionLevel", corruptionLevel); 
  gTools().ReadAttr(nn, "WasPretrained", wasPretrained); 


  if (wasPretrained==false) 
  {
    Log() << kFATAL << "The network which produced the weight file was not pretrained. " << Endl; 
  }

  SetInputWidth(inputWidth); 
  SetInputDepth(inputDepth); 
  SetInputHeight(inputHeight); 
  //SetOutputWidth
  SetBatchSize(batchSize); 
  SetBatchWidth(batchWidth); 
  SetBatchHeight(batchHeight); 
  SetBatchDepth(batchDepth); 
  SetInitialization(initialization); 
  SetLossFunction(lossFunction); 
  SetRegularization(regularization); 
  SetWeightDecay(weightDecay); 
  SetDropoutProbability(dropoutProbability); 
  SetCorruptionLevel(corruptionLevel); 

  int currentLayerType, currentActivationFunction, currentVisibleUnits, currentHiddenUnits; 
  Matrix_t weights, biases; 
  auto layer = gTools().xmlengine().GetChild(nn, "Layer"); 
  
  for (Int_t i = 0; i<depth; i++) 
  {
    gTools().ReadAttr(layer, "LayerType", currentLayerType); 
    gTools().ReadAttr(layer, "ActivationFunction", currentActivationFunction);
    gTools().ReadAttr(layer, "VisibleUnits", currentVisibleUnits); 
    gTools().ReadAttr(layer, "HiddenUnits", currentHiddenUnits);
    auto internalLayer = gTools().xmlengine().GetChild(layer, "InternalLayer"); 
    ReadMatrixXML(internalLayer, "Weights", weights);
    ReadMatrixXML(internalLayer, "Biases",  biases);
    //for (Int_t j=0; j< /*something*/; j++)
    switch(currentLayerType) 
    {
      case 0: 
        // Whatever layer it is ... 
      case 1: 
        AddCorruptionLayer(currentVisibleUnits, currentHiddenUnits, dropoutProbability, corruptionLevel); 
      case 2: 
        AddCompressionLayer(currentVisibleUnits, currentHiddenUnits, dropoutProbability, currentActivationFunction, weights, biases); 
      case 3: 
        AddReconstructionLayer(currentVisibleUnits, currentHiddenUnits, 0.1, currentActivationFunction, weights, biases, dropoutProbability, corruptionLevel); 
    }

  }
  gTools().xmlengine().FreeDoc(doc);
  }
}

template <typename Architecture_t, typename Layer_t>
auto TDeepAutoEncoder<Architecture_t, Layer_t>::ReadMatrixXML(void *xml,
                                     const char *name,
                                     TMatrixT<Double_t> &X) -> void
{
   void *matrixXML = gTools().GetChild(xml, name);
   size_t rows, cols;
   gTools().ReadAttr(matrixXML, "rows", rows);
   gTools().ReadAttr(matrixXML, "cols", cols);

   const char * matrixString = gTools().xmlengine().GetNodeContent(matrixXML);
   std::stringstream matrixStringStream(matrixString);

   for (size_t i = 0; i < rows; i++)
   {
      for (size_t j = 0; j < cols; j++)
      {
         matrixStringStream >> X(i,j);
      }
   }
}

} // namespace DAE
} // namespace DNN
} // namespace TMVA

#endif
