// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_reader_tests.hpp"
#include <string>

TEST_F(NGraphReaderTests, MaxPool2DDefault) {
  std::string model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,32,32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="MaxPool" version="opset1">
			<data kernel="2,2" pads_begin="0,0" pads_end="0,0" rounding_type="floor" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>31</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>31</dim>
					<dim>31</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
        )V0G0N";

  std::string modelV7 = R"V0G0N(
<net name="model" version="7">
	<layers>
		<layer id="0" name="x" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Pooling" version="opset1">
			<data exclude-pad="true" kernel="2,2" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="floor" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>31</dim>
					<dim>31</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>)V0G0N";
  compareIRs(model, modelV7, 0);
}

TEST_F(NGraphReaderTests, MaxPool2DNonDefaultRoundingType) {
  std::string model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,1,4,4"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="MaxPool" version="opset1">
			<data kernel="3,3" pads_begin="0,0" pads_end="0,0" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
        )V0G0N";

  std::string modelV7 = R"V0G0N(
<net name="model" version="7">
	<layers>
		<layer id="0" name="x" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Pooling" version="opset1">
			<data exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="ceil" strides="2,2"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>1</dim>
					<dim>4</dim>
					<dim>4</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>2</dim>
					<dim>2</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>)V0G0N";
  compareIRs(model, modelV7, 0);
}

TEST_F(NGraphReaderTests, MaxPool2DNonDefaultPads) {
  std::string model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,28,28"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="MaxPool" version="opset1">
			<data kernel="3,3" pads_begin="2,2" pads_end="2,2" rounding_type="floor" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
        )V0G0N";

  std::string modelV7 = R"V0G0N(
<net name="model" version="7">
	<layers>
		<layer id="0" name="x" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Pooling" version="opset1">
			<data exclude-pad="true" kernel="3,3" pads_begin="2,2" pads_end="2,2" pool-method="max" rounding_type="floor" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>28</dim>
					<dim>28</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>30</dim>
					<dim>30</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>)V0G0N";
  compareIRs(model, modelV7, 0);
}

TEST_F(NGraphReaderTests, MaxPool2DNonDefaultAutoPadSameUpper) {
  std::string model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,32,32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="MaxPool" version="opset1">
			<data auto_pad="same_upper" kernel="2,2" pads_begin="0,0" pads_end="1,1" rounding_type="ceil" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>
        )V0G0N";

  std::string modelV7 = R"V0G0N(
<net name="model" version="7">
	<layers>
		<layer id="0" name="x" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Pooling" version="opset1">
			<data auto_pad="same_upper" exclude-pad="true" kernel="2,2" pads_begin="0,0" pads_end="1,1" pool-method="max" rounding_type="ceil" strides="1,1"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>)V0G0N";
  compareIRs(model, modelV7, 0);
}

TEST_F(NGraphReaderTests, MaxPool2DNonDefaultStrides) {
  std::string model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,3,32,32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="MaxPool" version="opset1">
			<data kernel="5,5" pads_begin="0,0" pads_end="0,0" rounding_type="floor" strides="3,3"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
	</edges>
</net>)V0G0N";

  std::string modelV7 = R"V0G0N(
<net name="model" version="7">
	<layers>
		<layer id="0" name="x" type="Input" version="opset1">
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Pooling" version="opset1">
			<data exclude-pad="true" kernel="5,5" pads_begin="0,0" pads_end="0,0" pool-method="max" rounding_type="floor" strides="3,3"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>3</dim>
					<dim>32</dim>
					<dim>32</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
	</edges>
</net>)V0G0N";
  compareIRs(model, modelV7, 0);
}
